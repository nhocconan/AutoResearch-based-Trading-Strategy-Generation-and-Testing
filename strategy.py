#!/usr/bin/env python3
"""
Experiment #077: 12h KAMA Adaptive Trend with Daily HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than EMA/HMA, performing well in both trending and ranging conditions.
This is critical for 2025 bear/range market where pure trend followers failed.
Combine KAMA slope changes with Daily HMA trend bias + simple RSI filter.
Keep entry conditions LOOSE to ensure 10+ trades (learning from 0-trade failures).
Position sizing: 0.25 entry, 0.125 at 1.5R profit, 2.5*ATR trailing stoploss.
12h timeframe reduces noise vs lower TFs while maintaining trade frequency.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_v1"
timeframe = "12h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - moves fast in trends, slow in noise.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    ER = Change / Sum of absolute changes over er_period
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change in price
    change = np.abs(close_s.diff())
    change.iloc[0] = 0
    
    # Sum of absolute changes over er_period
    sum_changes = change.rolling(window=er_period, min_periods=er_period).sum()
    
    # Net change over er_period
    net_change = np.abs(close_s - close_s.shift(er_period))
    
    # Efficiency Ratio (ER)
    er = net_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0)
    er = er.clip(0, 1)
    
    # Smoothing Constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # KAMA slope (direction change detection)
    kama_slope = np.zeros(n)
    for i in range(1, n):
        kama_slope[i] = kama[i] - kama[i-1]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Daily trend filter (HTF) - price relative to Daily HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA slope signals (direction change)
        kama_turning_up = kama_slope[i] > 0 and kama_slope[i-1] <= 0
        kama_turning_down = kama_slope[i] < 0 and kama_slope[i-1] >= 0
        
        # KAMA trend state
        kama_uptrend = kama_slope[i] > 0
        kama_downtrend = kama_slope[i] < 0
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter (simple, not too strict)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (LOOSE to ensure trades)
        # Condition 1: KAMA turns up + Daily bullish + RSI bullish
        if kama_turning_up and daily_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        # Condition 2: Price crosses above KAMA + Daily bullish + KAMA uptrend
        elif price_above_kama and daily_bullish and kama_uptrend and rsi_bullish:
            # Check if just crossed (was below before)
            if i > 0 and close[i-1] <= kama[i-1]:
                new_signal = SIZE_ENTRY
        # Condition 3: Strong KAMA uptrend + Daily bullish (momentum continuation)
        elif kama_uptrend and daily_bullish and kama_slope[i] > atr[i] * 0.5:
            # Only if not already in strong uptrend for too long (avoid late entries)
            if i > 5 and np.mean(kama_slope[i-5:i]) > 0:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: KAMA turns down + Daily bearish + RSI bearish
        if kama_turning_down and daily_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 2: Price crosses below KAMA + Daily bearish + KAMA downtrend
        elif price_below_kama and daily_bearish and kama_downtrend and rsi_bearish:
            # Check if just crossed (was above before)
            if i > 0 and close[i-1] >= kama[i-1]:
                new_signal = -SIZE_ENTRY
        # Condition 3: Strong KAMA downtrend + Daily bearish (momentum continuation)
        elif kama_downtrend and daily_bearish and kama_slope[i] < -atr[i] * 0.5:
            # Only if not already in strong downtrend for too long
            if i > 5 and np.mean(kama_slope[i-5:i]) < 0:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals