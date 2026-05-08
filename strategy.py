#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 level, 1d EMA34 rising, volume > 2x average
# Short when price breaks below Camarilla S3 level, 1d EMA34 falling, volume > 2x average
# Uses Camarilla pivot levels from daily data for institutional-grade support/resistance
# Targets 50-150 total trades over 4 years (12-38/year) for optimal balance of edge and frequency

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Camarilla levels: R4 = C + (H-L)*1.500, R3 = C + (H-L)*1.250, R2 = C + (H-L)*1.166, R1 = C + (H-L)*1.083
    #                 PP = (H+L+C)/3, S1 = C - (H-L)*1.083, S2 = C - (H-L)*1.166, S3 = C - (H-L)*1.250, S4 = C - (H-L)*1.500
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points (use previous day's data to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN since we don't have previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.250
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.250
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data once for EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # start after first bar to allow for previous day data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3, 1d uptrend, volume spike
            if high_val > r3_level and ema34_val > ema34_1d_aligned[i-1] and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3, 1d downtrend, volume spike
            elif low_val < s3_level and ema34_val < ema34_1d_aligned[i-1] and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or 1d trend turns down
            if low_val < s3_level or ema34_val < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or 1d trend turns up
            if high_val > r3_level or ema34_val > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals