#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA(34) trend filter and ATR-based stoploss.
Long when price breaks above 20-day high AND 1w close > EMA(34). Short when breaks below 20-day low AND 1w close < EMA(34).
Uses ATR(14) for dynamic stoploss: exit long if price drops 2*ATR from entry, exit short if price rises 2*ATR from entry.
Volume confirmation: current volume > 1.5 * 20-day average volume.
Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) via 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) on 1d for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar: no previous close
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate Donchian channels (20-day high/low) from previous 1d bar
    # Using rolling window on previous day's high/low to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    
    # 20-period rolling max/min on previous day's data
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned via shift)
    donchian_high_aligned = donchian_high  # already aligned to 1d bars
    donchian_low_aligned = donchian_low    # already aligned to 1d bars
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    # Warmup: max of 1w EMA(34), ATR(14), Donchian(20), volume MA(20)
    start_idx = max(34, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_14_aligned[i]
        trend_up = close_val > ema_34_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirm AND 1w uptrend
            long_signal = (close_val > donchian_high_aligned[i]) and vol_conf and trend_up
            
            # Short: price breaks below Donchian low AND volume confirm AND 1w downtrend
            short_signal = (close_val < donchian_low_aligned[i]) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price drops 2*ATR below entry price (stoploss)
            # 2. Price breaks below Donchian low (failed breakout)
            # 3. 1w trend flips down
            if (close_val < entry_price - 2.0 * atr_val) or \
               (close_val < donchian_low_aligned[i]) or \
               (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price rises 2*ATR above entry price (stoploss)
            # 2. Price breaks above Donchian high (failed breakdown)
            # 3. 1w trend flips up
            if (close_val > entry_price + 2.0 * atr_val) or \
               (close_val > donchian_high_aligned[i]) or \
               (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0