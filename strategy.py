#!/usr/bin/env python3
"""
EXPERIMENT #007 - Bollinger-Keltner Squeeze Breakout (1h)
==========================================================
Hypothesis: Volatility contraction (squeeze) followed by expansion creates high-probability
breakout opportunities. BB inside KC indicates low volatility regime. When price breaks
out of squeeze with trend filter (SMA200), captures explosive moves while avoiding
counter-trend traps that killed RSI_SMA200 and KAMA strategies.

Why this should beat Supertrend_4h (Sharpe=0.197):
- Captures volatility explosions (crypto's main alpha source)
- SMA200 filter prevents counter-trend trades (fixes RSI failure)
- 1h timeframe = more opportunities than 4h, cleaner than 15m
- Squeeze detection reduces whipsaws during choppy periods
- Conservative sizing (0.30) controls drawdown better than failed strategies

Key improvements over baseline:
- Volatility regime detection (squeeze vs expansion)
- Trend filter avoids 2022-style crash losses
- Discrete signal levels (0, ±0.30) minimize churn costs
- Proper min_periods on all rolling calculations
"""

import numpy as np
import pandas as pd

name = "bb_kc_squeeze_1h_v1"
timeframe = "1h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === BOLLINGER BANDS (20, 2.0) ===
    bb_period = 20
    bb_mult = 2.0
    
    bb_sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_sma + bb_mult * bb_std
    bb_lower = bb_sma - bb_mult * bb_std
    
    # === KELTNER CHANNELS (20, 1.5 ATR) ===
    kc_period = 20
    kc_mult = 1.5
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    kc_atr = pd.Series(tr).rolling(window=kc_period, min_periods=kc_period).mean().values
    kc_sma = pd.Series(close).rolling(window=kc_period, min_periods=kc_period).mean().values
    kc_upper = kc_sma + kc_mult * kc_atr
    kc_lower = kc_sma - kc_mult * kc_atr
    
    # === SMA200 TREND FILTER ===
    sma200_period = 200
    sma200 = pd.Series(close).rolling(window=sma200_period, min_periods=sma200_period).mean().values
    
    # === SQUEEZE DETECTION ===
    # Squeeze = BB inside KC (volatility contraction)
    squeeze = np.zeros(n, dtype=bool)
    for i in range(max(bb_period, kc_period, sma200_period), n):
        if np.isnan(bb_upper[i]) or np.isnan(kc_upper[i]):
            continue
        # BB upper < KC upper AND BB lower > KC lower
        squeeze[i] = (bb_upper[i] < kc_upper[i]) and (bb_lower[i] > kc_lower[i])
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for drawdown control
    
    # Track squeeze state for breakout detection
    in_squeeze = False
    squeeze_start_idx = -1
    
    for i in range(max(bb_period, kc_period, sma200_period), n):
        if np.isnan(bb_upper[i]) or np.isnan(sma200[i]):
            continue
        
        # Detect squeeze start
        if squeeze[i] and not in_squeeze:
            in_squeeze = True
            squeeze_start_idx = i
        
        # Detect squeeze end (breakout)
        if in_squeeze and not squeeze[i]:
            in_squeeze = False
            
            # Check breakout direction
            breakout_long = close[i] > bb_upper[i]
            breakout_short = close[i] < bb_lower[i]
            
            # Apply trend filter
            above_sma200 = close[i] > sma200[i]
            below_sma200 = close[i] < sma200[i]
            
            # Long signal: breakout up + above SMA200
            if breakout_long and above_sma200:
                signals[i] = SIZE
            # Short signal: breakout down + below SMA200
            elif breakout_short and below_sma200:
                signals[i] = -SIZE
            # No signal if counter-trend breakout
            else:
                signals[i] = 0.0
        
        # Hold position during squeeze if already positioned
        elif in_squeeze:
            signals[i] = signals[i-1] if i > 0 else 0.0
        # Outside squeeze - maintain trend bias
        else:
            if close[i] > sma200[i]:
                signals[i] = SIZE * 0.5  # Reduced size outside squeeze
            elif close[i] < sma200[i]:
                signals[i] = -SIZE * 0.5
            else:
                signals[i] = 0.0
    
    # Apply discrete signal levels to reduce churn
    for i in range(n):
        if signals[i] > 0.15:
            signals[i] = SIZE
        elif signals[i] < -0.15:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals