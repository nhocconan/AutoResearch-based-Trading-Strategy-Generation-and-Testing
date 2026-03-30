#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian + Volume + HMA(21) 1d Trend + Choppiness

HYPOTHESIS: Donchian(20) breakout captures institutional momentum moves.
By requiring:
  1. Price breaks 4h Donchian(20) high/low
  2. Volume confirms breakout (>1.8x 20-avg)
  3. 1d HMA(21) confirms trend direction
  4. Choppiness < 50 (trending, not ranging)

This targets 100-150 total trades over 4 years with high win rate.

Entry: Donchian breakout + vol spike + HMA trend aligned + chop < 50
Exit: Opposite Donchian touch OR 2.5ATR stoploss
Size: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma21_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA of period/2 * 2 - WMA of period
    half = period // 2
    wma_half = pd.Series(data).rolling(window=half, min_periods=half).apply(
        lambda x: np.dot(x, np.arange(half)) / (half * (half - 1) / 2), raw=True
    )
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(period)) / (period * (period - 1) / 2), raw=True
    )
    sqrt_period = int(np.sqrt(period))
    wma_sqrt = pd.Series(data).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.dot(x, np.arange(sqrt_period)) / (sqrt_period * (sqrt_period - 1) / 2), raw=True
    )
    
    hma = 2 * wma_half - wma_full
    hma = hma.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values < 38.2 = trending, > 61.8 = ranging"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            idx = i - j
            tr = max(high[idx] - low[idx], abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
            sum_tr += tr
        
        highest_high = max(high[i-period+1:i+1])
        lowest_low = min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10:
            chop[i] = 100 * (np.log(sum_tr) / np.log(range_val * period)) if range_val > 0 else 50
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === Local 4h indicators ===
    # Donchian channels (20 bars = 5 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    entry_donch_high = 0.0
    entry_donch_low = 0.0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Choppiness filter: < 50 = trending (can enter)
        # > 61.8 = ranging (skip)
        is_trending = chop[i] < 50.0
        
        # === TREND DIRECTION (1d HMA21) ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        is_bullish = price_above_hma
        is_bearish = not price_above_hma
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # Donchian breakout yesterday (confirmed - no look-ahead)
        # We use shifted values from previous bar
        prev_donch_high = donchian_high[i - 1] if i > 0 else 0
        prev_donch_low = donchian_low[i - 1] if i > 0 else 0
        
        # Check if price broke out yesterday
        prev_close = close[i - 1]
        bull_breakout = prev_close > prev_donch_high
        bear_breakout = prev_close < prev_donch_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bull breakout + vol spike + trend aligned + trending market ===
            if bull_breakout and vol_spike and is_bullish and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Bear breakout + vol spike + trend aligned + trending market ===
            if bear_breakout and vol_spike and is_bearish and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD = 4 bars (1 day) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Exit on opposite Donchian touch
            if position_side > 0 and low[i] <= prev_donch_low:
                desired_signal = 0.0
            if position_side < 0 and high[i] >= prev_donch_high:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                entry_donch_high = prev_donch_high
                entry_donch_low = prev_donch_low
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals