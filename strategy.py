#!/usr/bin/env python3
"""
EXPERIMENT #006 - EMA Crossover with Weekly Trend + RSI Momentum (1d)
======================================================================
Hypothesis: Daily EMA(12/26) crossovers capture trend changes earlier than 
Donchian breakouts, with less whipsaw when filtered by weekly EMA(50) trend 
direction and RSI momentum confirmation. ATR trailing stop protects capital.

Key features:
- Primary TF: 1d (daily candles)
- HTF filter: 1w EMA(50) for major trend direction
- Entry: EMA(12) crosses EMA(26) + RSI(14) confirms momentum
- Filter: Weekly trend must align with crossover direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.28 discrete levels (28% of capital)
- Take profit: Reduce to half at 2R, continue trailing

This differs from Donchian by using MA crossovers (smoother, earlier signals)
instead of pure price breakouts (later, but cleaner).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_crossover_weekly_rsi_1d_v1"
timeframe = "1d"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    ema_1w = calculate_ema(df_1w['close'].values, 50)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    ema_fast = calculate_ema(close, 12)
    ema_slow = calculate_ema(close, 26)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for weekly EMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend = 1 if close[i] > ema_1w_aligned[i] else -1
        
        # EMA crossover detection
        crossover_signal = 0
        if i > 0:
            # Long crossover: fast crosses above slow
            if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                crossover_signal = 1
            # Short crossover: fast crosses below slow
            elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                crossover_signal = -1
        
        # RSI momentum filter (confirm trend direction)
        rsi_valid = False
        if crossover_signal == 1 and rsi[i] > 50:  # Long needs RSI > 50
            rsi_valid = True
        elif crossover_signal == -1 and rsi[i] < 50:  # Short needs RSI < 50
            rsi_valid = True
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if crossover_signal != 0:
            # Crossover must align with weekly trend and RSI confirms
            if crossover_signal == weekly_trend and rsi_valid:
                target_signal = SIZE * crossover_signal
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR_entry
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            # Continue trailing stop (position_side stays the same)
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Check for position reversal
                if position_side != 0 and ((position_side == 1 and target_signal < 0) or (position_side == -1 and target_signal > 0)):
                    # Exit current position and enter opposite on same bar
                    signals[i] = target_signal  # Directly set to new position
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    entry_atr = atr[i]
                    profit_target_hit = False
                else:
                    signals[i] = target_signal
                    if position_side == 0:
                        # New entry
                        position_side = 1 if target_signal > 0 else -1
                        highest_since_entry = close[i]
                        lowest_since_entry = close[i]
                        entry_price = close[i]
                        entry_atr = atr[i]
                        profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals