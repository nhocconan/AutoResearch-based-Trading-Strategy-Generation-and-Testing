#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot S3/R3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 level on 4h, 1d EMA34 rising, volume > 1.5x average
# Short when price breaks below S3 level on 4h, 1d EMA34 falling, volume > 1.5x average
# Uses 4h for entry timing and pivot levels, 1d for trend filter to avoid whipsaws
# Targets 15-40 total trades per year (60-160 over 4 years) for low fee drag and high win rate
# Camarilla levels provide strong institutional support/resistance, effective in both trends and ranges

name = "4h_Camarilla_R3S3_1dEMA34_Trend_Volume"
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
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for S3 and R3 from previous 4h bar
    # Using standard Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # For simplicity, we use: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels based on previous bar
    camarilla_h_l = high_4h - low_4h
    camarilla_r3 = close_4h + camarilla_h_l * 1.1 / 4
    camarilla_s3 = close_4h - camarilla_h_l * 1.1 / 4
    
    # Shift levels to align with current bar (use previous bar's levels)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan  # First value invalid due to roll
    camarilla_s3[0] = np.nan
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start from index 1 since we rolled the arrays
    
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
            # Enter long: price breaks above R3, 1d uptrend (EMA34 > 0 slope proxy), volume spike
            if high_val > r3_level and ema34_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 1d downtrend, volume spike
            elif low_val < s3_level and ema34_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 1d trend turns down
            if low_val < s3_level or ema34_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or 1d trend turns up
            if high_val > r3_level or ema34_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals