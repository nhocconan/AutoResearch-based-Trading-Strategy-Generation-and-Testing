#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike_v2
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 1d with 1w EMA50 trend filter and volume confirmation (2.0x median).
Only trade in direction of 1w EMA50 trend to reduce whipsaws. Target: 7-25 trades/year on 1d timeframe.
Works in bull/bear by following 1w EMA50 trend and filtering counter-trend breaks. Uses ATR-based trailing stop.
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
    
    # Get 1w data for HTF trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1w OHLC
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    camarilla_r1 = prev_close_1w + 1.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_s1 = prev_close_1w - 1.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_r2 = prev_close_1w + 2.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_s2 = prev_close_1w - 2.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_r3 = prev_close_1w + 3.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_s3 = prev_close_1w - 3.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_r4 = prev_close_1w + 4.000/6 * (prev_high_1w - prev_low_1w)
    camarilla_s4 = prev_close_1w - 4.000/6 * (prev_high_1w - prev_low_1w)
    
    # Align HTF indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
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
    
    # Warmup: max of 1w EMA (50), volume median (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend (close > 1w EMA50)
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_1w_val)
            
            # Short: break below S1 with volume spike, and downtrend (close < 1w EMA50)
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_1w_val)
            
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

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0