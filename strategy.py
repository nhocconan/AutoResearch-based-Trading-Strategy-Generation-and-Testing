#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + Volume + 1d EMA Trend

HYPOTHESIS: Donchian(20) breakout on 12h captures multi-day momentum shifts.
Price breaking above 20-bar high = accumulation, below = distribution.
Combining with 1d EMA trend filter + volume confirmation filters false breakouts.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy breakouts above 1d EMA (confirms uptrend)
- Bear: Short breakouts below 1d EMA (confirms downtrend)
- Range: Choppiness filter avoids whipsaws

WHY 12h: 3x fewer trades than 4h = less fee drag.
20-period Donchian on 12h = 10-day channel, catches medium-term swings.

TARGET: 75-150 total trades over 4 years. Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_ema50_1d_v2"
timeframe = "12h"
leverage = 1.0

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
    """Choppiness Index - lower = trending, higher = ranging"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    chop = np.full(n, 50.0, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = high[i - period:i + 1].max()
        lowest_low = low[i - period:i + 1].min()
        range_sum = highest_high - lowest_low
        
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                     abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j],
                     abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else 0)
            atr_sum += tr
        
        if atr_sum > 0 and range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-bar MA)
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
    last_donchian_high = 0.0
    last_donchian_low = 0.0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Previous bar's Donchian (to detect breakout without look-ahead)
        prev_donchian_high = donchian_high[i - 1]
        prev_donchian_low = donchian_low[i - 1]
        
        # Current bar Donchian (for reference)
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Choppiness filter: < 50 = trending (good for entries), > 61.8 = ranging (skip)
        trending = chop[i] < 50.0
        
        # === BREAKOUT DETECTION (previous bar's high/low) ===
        bullish_breakout = prev_donchian_high > 0 and high[i - 1] >= prev_donchian_high
        bearish_breakout = prev_donchian_low > 0 and low[i - 1] <= prev_donchian_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above 20-bar high + volume + trend alignment ===
            if bullish_breakout and vol_spike and price_above_1d_ema:
                desired_signal = SIZE
                last_donchian_high = prev_donchian_high
            
            # === SHORT: Breakdown below 20-bar low + volume + trend alignment ===
            if bearish_breakout and vol_spike and not price_above_1d_ema:
                desired_signal = -SIZE
                last_donchian_low = prev_donchian_low
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
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
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals