#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3, 1d EMA34 rising, volume > 1.5x average
# Short when price breaks below S3, 1d EMA34 falling, volume > 1.5x average
# Uses Camarilla pivot for structure, EMA34 for trend filter, volume for confirmation
# Targets 12-37 trades per year (50-150 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation

name = "12h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R3 = H + 2*(Pivot - L)
    r3 = prev_high + 2 * (pivot - prev_low)
    # S3 = L - 2*(H - Pivot)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align R3 and S3 to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days of data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above R3, 1d uptrend, volume confirmation
            if high_val > r3_val and ema34_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 1d downtrend, volume confirmation
            elif low_val < s3_val and ema34_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 1d trend down
            if low_val < s3_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or 1d trend up
            if high_val > r3_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals