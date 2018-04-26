// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CC_POSITION_H_
#define CC_POSITION_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/group.h"
#include "cc/inline_vector.h"
#include "cc/stone.h"

namespace minigo {

// BoardVisitor visits points on the board only once.
// A simple example that visits all points on the board only once:
//   BoardVisitor bv;
//   bv.Begin()
//   bv.Visit(0);
//   while (!bv.Done()) {
//     Coord coord = bv.Next();
//     std::cout << "Visiting " << coord << "\n";
//     for (auto neighbor_coord : GetNeighbors(coord)) {
//       bv.Visit(neighbor_coord);
//     }
//   }
//
// Points are visited in the order that they are passed to Visit for the first
// time.
class BoardVisitor {
 public:
  BoardVisitor() = default;

  // Starts a new visit around the board.
  void Begin() {
    assert(Done());
    if (epoch_++ == 0) {
      memset(visited_.data(), 0, sizeof(visited_));
    }
  }

  // Returns true when there are no more points to visit.
  bool Done() const { return stack_.empty(); }

  // Returns the coordinates of the next point in the queue to visit.
  Coord Next() {
    auto c = stack_.back();
    stack_.pop_back();
    return c;
  }

  // If this is the first time Visit has been passed coordinate c since the most
  // recent call to Begin, Visit pushes the coordinate onto its queue of points
  // to visit and returns true. Otherwise, Visit returns false.
  bool Visit(Coord c) {
    if (visited_[c] != epoch_) {
      visited_[c] = epoch_;
      stack_.push_back(c);
      return true;
    }
    return false;
  }

 private:
  inline_vector<Coord, kN * kN> stack_;
  std::array<uint8_t, kN * kN> visited_;
  uint8_t epoch_ = 0;
};

// GroupVisitor simply keeps track of which groups have been visited since the
// most recent call to Begin. Unlike BoardVisitor, it does not keep a pending
// queue of groups to visit.
class GroupVisitor {
 public:
  GroupVisitor() = default;

  void Begin() {
    if (epoch_++ == 0) {
      memset(visited_.data(), 0, sizeof(visited_));
    }
  }

  bool Visit(GroupId id) {
    if (visited_[id] != epoch_) {
      visited_[id] = epoch_;
      return true;
    }
    return false;
  }

 private:
  uint8_t epoch_ = 0;
  std::array<uint8_t, Group::kMaxNumGroups> visited_;
};

// Position represents a single board position.
// It tracks the stones on the board and their groups, and contains the logic
// for removing groups with no remaining liberties and merging neighboring
// groups of the same color.
//
// Since the MCTS code makes a copy of the board position for each expanded
// node in the tree, we aim to keep the data structures as compact as possible.
// This is in tension with our other aim of avoiding heap allocations where
// possible, which means we have to preallocate some pools of memory. In
// particular, the BoardVisitor and GroupVisitor classes that Position uses to
// update its internal state are relatively large compared to the board size
// (even though we're only talking a couple of kB in total. Consequently, the
// caller of the Position code must pass pointers to previously allocated
// instances of BoardVisitor and GroupVisitor. These can then be reused by all
// instances of the Position class.
class Position {
 public:
  Position(BoardVisitor* bv, GroupVisitor* gv, float komi, Color to_play,
           int n = 0);

  // Copies the position's state from another instance, while preserving the
  // BoardVisitor and GroupVisitor it was constructed with.
  Position(BoardVisitor* bv, GroupVisitor* gv, const Position& other);

  Position(const Position&) = default;
  Position& operator=(const Position&) = default;

  void PlayMove(Coord c, Color color = Color::kEmpty);

  // Adds the stone to the board.
  // Removes newly surrounded opponent groups.
  // Updates liberty counts of remaining groups.
  // Updates num_captures_.
  // If the move captures a single stone, sets ko_ to the coordinate of that
  // stone. Sets ko_ to kInvalid otherwise.
  void AddStoneToBoard(Coord c, Color color);

  const std::array<int, 2>& num_captures() const { return num_captures_; }

  // Calculates the score from B perspective. If W is winning, score is
  // negative.
  float CalculateScore();

  // Returns true if playing this move is legal.
  bool IsMoveLegal(Coord c) const;

  std::string ToSimpleString() const;
  std::string ToGroupString() const;
  std::string ToPrettyString() const;

  Color to_play() const { return to_play_; }
  Coord previous_move() const { return previous_move_; }
  const std::array<Stone, kN * kN>& stones() const { return stones_; }
  int n() const { return n_; }
  bool is_game_over() const { return num_consecutive_passes_ >= 2; }

  // The following methods are protected to enable direct testing by unit tests.
 protected:
  // Returns the Group of the stone at the given coordinate. Used for testing.
  Group GroupAt(Coord c) const {
    auto s = stones_[c];
    return s.empty() ? Group() : groups_[s.group_id()];
  }

  // Returns color C if the position at idx is empty and surrounded on all
  // sides by stones of color C.
  // Returns Color::kEmpty otherwise.
  Color IsKoish(Coord c) const;

  // Returns true if playing this move is suicidal.
  bool IsMoveSuicidal(Coord c, Color color) const;

 private:
  // Play a pass move.
  void PassMove();

  // Removes the group with a stone at the given coordinate from the board,
  // updating the liberty counts of neighboring groups.
  void RemoveGroup(Coord c);

  // Merge neighboring groups of the same color as the stone at coordinate c
  // into that stone's group. Called when a stone is placed on the board that
  // has two or more distinct neighboring groups of the same color.
  void MergeGroup(Coord c);

  // Returns true if the point at coordinate c neighbors the given group.
  bool HasNeighboringGroup(Coord c, GroupId group_id) const;

  std::array<Stone, kN * kN> stones_;
  BoardVisitor* board_visitor_;
  GroupVisitor* group_visitor_;
  GroupPool groups_;

  Color to_play_;
  Coord previous_move_ = Coord::kInvalid;
  Coord ko_ = Coord::kInvalid;

  // Number of captures for (B, W).
  std::array<int, 2> num_captures_ = std::array<int, 2>{0, 0};

  int n_;
  int num_consecutive_passes_ = 0;
  float komi_;
};

}  // namespace minigo

#endif  // CC_POSITION_H_
