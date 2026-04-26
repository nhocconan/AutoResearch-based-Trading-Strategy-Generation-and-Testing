#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and volume spike (1.8x median). 
Uses 4h timeframe for HTF trend to avoid 1h noise while maintaining precise entry timing.
Adds 08-20 UTC session filter to reduce off-hour whipsaws.
Target: 20-40 trades/year on 1h (80-160 total over 4 years) to minimize fee drag.
Designed to work in both bull and bear markets via trend filter + volatility-based stops.
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
    
    # Get 4h data for HTF trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 4h OHLC
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_r1 = prev_close_4h + 1.000/6 * (prev_high_4h - prev_low_4h)
    camarilla_s1 = prev_close_4h - 1.000/6 * (prev_high_4h - prev_low_4h)
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: 1.8x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 4h, volume median (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            # Hold current position or flat if outside session
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 1.8 * vol_median_val) and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 1.8 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0