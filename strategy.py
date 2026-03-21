#!/usr/bin/env python3
"""
EXPERIMENT #009 - KAMA Adaptive Trend + MACD Momentum + ADX Filter (1h primary, 4h HTF)
=======================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than HMA/EMA, reducing whipsaws in choppy markets. Combined with MACD histogram for 
momentum confirmation and ADX > 25 for trend strength filter, this should reduce 
false signals. 4h KAMA provides major trend alignment, 1h for entry timing.

Key differences from failed strategies:
- KAMA adapts ER (Efficiency Ratio) to market conditions (unlike fixed HMA/EMA)
- ADX > 25 filter ensures we only trade in trending markets (avoid chop)
- MACD histogram divergence for entry timing (not just crossover)
- Conservative position sizing: 0.25 base, discrete levels
- 2.5*ATR trailing stoploss with 2R take profit reduction

Primary TF: 1h | HTF: 4h KAMA(21) | Stoploss: 2.5*ATR | Size: 0.25-0.30
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_adx_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility using Efficiency Ratio (ER)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(period).abs()
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, period=21)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)  # auto shift(1) for completed bars
    
    # Calculate 1h indicators
    kama_1h = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, 14)
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    adx = calculate_adx(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    prev_histogram = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_1h[i]) or 
            np.isnan(atr[i]) or np.isnan(histogram[i]) or np.isnan(adx[i]) or 
            atr[i] == 0 or np.isnan(macd_line[i])):
            signals[i] = 0.0
            continue
        
        # 4h KAMA trend filter (HTF) - major trend direction
        htf_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        
        # 1h KAMA trend filter - local trend
        ltf_trend = 1 if close[i] > kama_1h[i] else -1
        
        # ADX trend strength filter - only trade when ADX > 25 (trending market)
        trend_strength_valid = adx[i] > 25
        
        # MACD histogram momentum - look for turning points
        histogram_positive = histogram[i] > 0
        histogram_negative = histogram[i] < 0
        histogram_turning_long = histogram[i] > 0 and prev_histogram <= 0
        histogram_turning_short = histogram[i] < 0 and prev_histogram >= 0
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF bullish + LTF bullish + ADX strong + MACD turning positive
        if htf_trend == 1 and ltf_trend == 1 and trend_strength_valid and histogram_turning_long:
            target_signal = SIZE
        
        # Short entry: HTF bearish + LTF bearish + ADX strong + MACD turning negative
        elif htf_trend == -1 and ltf_trend == -1 and trend_strength_valid and histogram_turning_short:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[entry_idx if 'entry_idx' in dir() else i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[entry_idx if 'entry_idx' in dir() else i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_idx = i
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and ltf_trend == -1:
                    # LTF trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and ltf_trend == 1:
                    # LTF trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
        
        # Store previous histogram for turning point detection
        prev_histogram = histogram[i]
    
    return signals