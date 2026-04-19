#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 12h trend filter and volume confirmation
# Long when: Price breaks above R3 (Camarilla resistance) with volume > 1.5x average and 12h EMA34 rising
# Short when: Price breaks below S3 (Camarilla support) with volume > 1.5x average and 12h EMA34 falling
# Exit when price returns to pivot point (PP) or reverses at next level
# Designed for ~25-40 trades/year per symbol with strong edge in both bull and bear markets
name = "4h_Camarilla_R3S3_Breakout_Volume_EMA34"
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
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_34_slope = ema_12h_34 - np.roll(ema_12h_34, 1)
    ema_12h_34_slope[0] = 0
    ema_12h_34_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_34_slope)
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    range_1d = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_r3 = camarilla_pp + (range_1d * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_34_slope_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_20[i]
        vol_current = volume[i]
        ema_slope = ema_12h_34_slope_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        
        if position == 0:
            # Long: Break above R3 with volume confirmation and bullish 12h trend
            if price > r3 and vol_current > 1.5 * vol_ma and ema_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume confirmation and bearish 12h trend
            elif price < s3 and vol_current > 1.5 * vol_ma and ema_slope < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Return to PP or bearish reversal at next level
            if price < pp or (price < r3 and ema_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Return to PP or bullish reversal at next level
            if price > pp or (price > s3 and ema_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals