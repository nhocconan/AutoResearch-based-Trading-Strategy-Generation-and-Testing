#!/usr/bin/env python3
"""
Experiment #051: 1h KAMA Adaptive Trend with 4h HMA Filter + Choppiness Regime
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) which adapts to market volatility
- fast in trends, slow in ranges. Combine with 4h HMA for trend direction filter.
Add Choppiness Index (CHOP) to avoid trading in ranging markets (CHOP>61.8 = skip).
RSI extremes ( <35 / >65 ) for entry timing within trend direction.
Wider 2.5*ATR stoploss to avoid premature exits in volatile 1h bars.
Conservative sizing (0.25) with discrete levels to minimize fee churn.
1h timeframe should generate 20-50 trades/year with better timing than 4h/12h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_chop_rsi_v1"
timeframe = "1h"
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
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    Efficiency Ratio (ER) = |close - close_n| / sum(|close_i - close_i-1|)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    er_num = np.abs(close - np.roll(close, er_period))
    er_num[:er_period] = np.abs(close[:er_period] - close[0])
    
    er_den = np.zeros(n)
    for i in range(er_period, n):
        er_den[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    er_den[:er_period] = er_den[er_period] if er_den[er_period] > 0 else 1.0
    
    er = np.where(er_den > 0, er_num / er_den, 0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (avoid trading)
    CHOP < 38.2 = trending market (good for trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh - ll > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 100.0
    
    chop[:period] = 100.0  # Default to ranging for warmup
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # KAMA for adaptive trend following
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF) - use previous completed 4h bar
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Choppiness filter - only trade when CHOP < 61.8 (trending market)
        is_trending = chop[i] < 61.8
        
        # KAMA crossover signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA flip detection
        kama_flip_long = (i > 0) and (kama_fast[i] > kama_slow[i]) and (kama_fast[i-1] <= kama_slow[i-1])
        kama_flip_short = (i > 0) and (kama_fast[i] < kama_slow[i]) and (kama_fast[i-1] >= kama_slow[i-1])
        
        # RSI entry timing
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY: Trending market + 4h bullish + KAMA flip or pullback
        if is_trending and trend_bullish:
            if kama_flip_long:
                new_signal = SIZE_ENTRY
            elif kama_bullish and rsi_oversold and rsi_rising:
                new_signal = SIZE_ENTRY
            elif kama_bullish and close[i] > kama_fast[i] and rsi[i] > 45:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Trending market + 4h bearish + KAMA flip or pullback
        if is_trending and trend_bearish:
            if kama_flip_short:
                new_signal = -SIZE_ENTRY
            elif kama_bearish and rsi_overbought and rsi_falling:
                new_signal = -SIZE_ENTRY
            elif kama_bearish and close[i] < kama_fast[i] and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest price since entry for trailing stop
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_since_entry - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    initial_risk = entry_price - (entry_price - 2.5 * atr[int(i)])
                    profit = close[i] - entry_price
                    if initial_risk > 0 and profit >= 2.0 * initial_risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price since entry for trailing stop
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_since_entry + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    initial_risk = (entry_price + 2.5 * atr[int(i)]) - entry_price
                    profit = entry_price - close[i]
                    if initial_risk > 0 and profit >= 2.0 * initial_risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals