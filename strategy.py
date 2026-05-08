#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2x 20-period average AND price > 1w EMA50.
# Short when price breaks below Camarilla S3 AND 1d volume > 2x 20-period average AND price < 1w EMA50.
# Exit when price crosses back inside the Camarilla H-L range.
# Uses 4h timeframe for better trade frequency control (target 20-50/year).
# Camarilla levels from daily pivot provide institutional support/resistance.
# Volume spike filters for institutional participation.
# Weekly EMA50 ensures alignment with higher timeframe trend.

name = "4h_Camarilla_R3S3_1dVolumeSpike_1wEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla calculation and volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Weekly data for EMA50 trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 (inner strong levels)
    # Using previous day's OHLC
    prev_close_d = np.roll(df_d['close'].values, 1)
    prev_high_d = np.roll(df_d['high'].values, 1)
    prev_low_d = np.roll(df_d['low'].values, 1)
    prev_close_d[0] = df_d['close'].values[0]  # First day uses same day
    prev_high_d[0] = df_d['high'].values[0]
    prev_low_d[0] = df_d['low'].values[0]
    
    # Camarilla R3 and S3
    camarilla_r3 = prev_close_d + (prev_high_d - prev_low_d) * 1.1 / 4
    camarilla_s3 = prev_close_d - (prev_high_d - prev_low_d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_d, camarilla_s3)
    
    # Daily volume filter: current volume > 2x 20-period average (institutional participation)
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_spike_d = volume_d > (2.0 * vol_ma20_d)
    volume_spike = align_htf_to_ltf(prices, df_d, volume_spike_d)
    
    # Weekly EMA50 for trend filter
    close_w = df_w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Trend filter: price above/below weekly EMA50
    price_above_ema50 = close > ema50_w_aligned
    price_below_ema50 = close < ema50_w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup for all indicators
    start_idx = max(20, 50)  # For volume MA20 and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema50_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, volume spike, price above weekly EMA50
            long_cond = (close[i] > camarilla_r3_aligned[i]) and volume_spike[i] and price_above_ema50[i]
            # Short conditions: price breaks below Camarilla S3, volume spike, price below weekly EMA50
            short_cond = (close[i] < camarilla_s3_aligned[i]) and volume_spike[i] and price_below_ema50[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S3 (opposite level)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R3 (opposite level)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals