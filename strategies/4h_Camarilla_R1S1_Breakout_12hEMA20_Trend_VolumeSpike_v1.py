#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA20_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 4h with 12h EMA20 trend filter and volume confirmation (2.0x median).
Only trade in direction of 12h EMA20 trend to reduce whipsaws. Target: 20-50 trades/year on 4h timeframe.
Works in bull/bear by following 12h EMA20 trend and filtering counter-trend breaks. Uses ATR-based trailing stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 12h OHLC
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    camarilla_r1 = prev_close_12h + 1.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_s1 = prev_close_12h - 1.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_r2 = prev_close_12h + 2.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_s2 = prev_close_12h - 2.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_r3 = prev_close_12h + 3.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_s3 = prev_close_12h - 3.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_r4 = prev_close_12h + 4.000/6 * (prev_high_12h - prev_low_12h)
    camarilla_s4 = prev_close_12h - 4.000/6 * (prev_high_12h - prev_low_12h)
    
    # Align HTF indicators to 4h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 12h EMA (20), volume median (20), ATR (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_20_12h_val = ema_20_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend (close > 12h EMA20)
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_20_12h_val)
            
            # Short: break below S1 with volume spike, and downtrend (close < 12h EMA20)
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_20_12h_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if high_val - low_val > 0:  # avoid division by zero
                if close_val < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if high_val - low_val > 0:  # avoid division by zero
                if close_val > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA20_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0