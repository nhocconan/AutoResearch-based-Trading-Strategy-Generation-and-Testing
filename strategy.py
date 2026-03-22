#!/usr/bin/env python3
"""
Experiment #554: 4h Primary + 12h HTF — KAMA Adaptive Trend + Fisher Reversals

Hypothesis: After 493 failed strategies, the pattern is clear:
- Complex regime switching (chop + ADX + volume) consistently FAILS
- HMA crossover alone is too laggy for 4h timeframe
- KAMA (Kaufman Adaptive) adapts to market noise automatically — no regime filter needed
- Fisher Transform catches reversals better than RSI in bear/range markets (2022, 2025)
- Simpler logic = MORE trades = better chance of positive Sharpe

This strategy uses:
1. 4h KAMA(21) for adaptive trend (ER-based, smooths in chop, fast in trends)
2. 12h KAMA(21) aligned for major trend bias (filter counter-trend)
3. Fisher Transform(9) for reversal entries: long when Fisher < -1.5 crossing up, short when > +1.5 crossing down
4. ATR(14) 2.5x trailing stop for all positions
5. Simple position sizing: 0.30 bull regime, 0.25 bear regime

Why this might beat Sharpe=0.435:
- KAMA adapts automatically — no chop/trend regime detection needed (failed 10+ times)
- Fisher Transform proven in bear markets (2022 crash, 2025 decline)
- 12h HTF filter prevents major counter-trend losses
- Simpler entry = more trades (target 30-50/year) = less 0-trade failure risk
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Position sizing: 0.25-0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_reversal_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER (trending) → fast SC, Low ER (choppy) → slow SC
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |close - close[period]| / sum(|close[i] - close[i-1]|)
    net_change = np.abs(close_s - close_s.shift(period))
    total_change = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = net_change / (total_change + 1e-10)
    
    # Smoothing Constant: SC = ER * (fast_SC - slow_SC) + slow_SC
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation (recursive)
    kama = np.zeros(n)
    kama[period] = close[period]  # initialize
    
    for i in range(period + 1, n):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + (sc.iloc[i] ** 2) * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
        else:
            # Normalize price to -1 to +1 range
            value = 0.66 * ((close[i] - lowest) / (highest - lowest + 1e-10) - 0.5) + 0.67 * fisher[i-1] if i > 0 else 0.0
            value = np.clip(value, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value + 1e-10)) + 0.5 * fisher[i-1] if i > 0 else 0.0
        
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF KAMA for major trend direction
    kama_12h_21 = calculate_kama(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # 4h KAMA for adaptive trend
    kama_4h_21 = calculate_kama(close, period=21)
    kama_4h_50 = calculate_kama(close, period=50)
    
    # Fisher Transform for reversals
    fisher, fisher_signal = calculate_fisher(close, period=9)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_BULL = 0.30
    POSITION_SIZE_BEAR = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crossover state
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_12h_21_aligned[i]):
            continue
        if np.isnan(kama_4h_21[i]) or np.isnan(kama_4h_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > kama_12h_21_aligned[i]
        bear_regime_12h = close[i] < kama_12h_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (KAMA adaptive) ===
        bull_regime_4h = close[i] > kama_4h_21[i]
        bear_regime_4h = close[i] < kama_4h_21[i]
        
        kama_4h_slope_bull = kama_4h_21[i] > kama_4h_50[i]
        kama_4h_slope_bear = kama_4h_21[i] < kama_4h_50[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Track previous state to avoid repeated signals
        curr_fisher_long = fisher[i] < -1.0
        curr_fisher_short = fisher[i] > 1.0
        
        # === ENTRY LOGIC — SIMPLIFIED ===
        new_signal = 0.0
        
        # LONG ENTRY: 12h bull + 4h bull + Fisher reversal OR pullback
        if bull_regime_12h and bull_regime_4h:
            # Fisher reversal entry (primary)
            if fisher_long_cross and not prev_fisher_long_signal:
                new_signal = POSITION_SIZE_BULL if kama_4h_slope_bull else POSITION_SIZE_BULL * 0.8
            # KAMA pullback entry (secondary - price touches KAMA in uptrend)
            elif close[i] <= kama_4h_21[i] * 1.005 and close[i] >= kama_4h_21[i] * 0.995:
                if not in_position or position_side < 0:
                    new_signal = POSITION_SIZE_BULL * 0.8
        
        # SHORT ENTRY: 12h bear + 4h bear + Fisher reversal OR rally
        elif bear_regime_12h and bear_regime_4h:
            # Fisher reversal entry (primary)
            if fisher_short_cross and not prev_fisher_short_signal:
                new_signal = -POSITION_SIZE_BEAR if kama_4h_slope_bear else -POSITION_SIZE_BEAR * 0.8
            # KAMA rally entry (secondary - price touches KAMA in downtrend)
            elif close[i] >= kama_4h_21[i] * 0.995 and close[i] <= kama_4h_21[i] * 1.005:
                if not in_position or position_side > 0:
                    new_signal = -POSITION_SIZE_BEAR * 0.8
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        # Update Fisher signal tracking
        prev_fisher_long_signal = curr_fisher_long
        prev_fisher_short_signal = curr_fisher_short
        
        signals[i] = new_signal
    
    return signals