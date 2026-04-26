#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume spike >2x median. Uses 4h trend for signal direction to avoid whipsaws, volume confirms institutional interest. Targets 15-35 trades/year by using HTF for direction and 1h only for entry timing. Added 08-20 UTC session filter to reduce noise trades. Fixed 0.20 position size to limit fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (R1, S1) from previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's values (shifted by 1)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev = np.roll(close_4h, 1)
    # First bar: use same values (will be filtered by min_periods later)
    high_4h_prev[0] = high_4h[0]
    low_4h_prev[0] = low_4h[0]
    close_4h_prev[0] = close_4h[0]
    
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_range = high_4h_prev - low_4h_prev
    r1 = close_4h_prev + camarilla_range * 1.1 / 12
    s1 = close_4h_prev - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h (no extra delay needed as they're based on completed 4h candles)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume spike: volume > 2x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 4h EMA, 20 for volume median
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_median_20[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R1 with volume spike, uptrend (close > EMA50_4h)
            long_entry = (close_val > r1_aligned[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price breaks below S1 with volume spike, downtrend (close < EMA50_4h)
            short_entry = (close_val < s1_aligned[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Camarilla (below S1)
            if close_val < ema_50_val or close_val < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Camarilla (above R1)
            if close_val > ema_50_val or close_val > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0