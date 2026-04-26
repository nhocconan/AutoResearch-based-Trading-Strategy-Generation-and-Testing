#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 4h breakout above/below weekly Camarilla R3/S3 levels in direction of 1w EMA34 trend, confirmed by volume spike (>2.0x 50-bar MA). Weekly trend filter avoids counter-trend trades in bear markets. Camarilla levels from higher timeframe provide institutional support/resistance. Volume confirmation reduces false breakouts. Designed for 20-50 trades/year (80-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1w trend while using Camarilla structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels on 1w (based on previous week's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as primary breakout levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (50 for vol, 34 for ema)
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1w = close_val > ema_34_val
        bearish_1w = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R3/S3 levels in trend direction with volume
        # Camarilla R3 = resistance level, break above = long
        # Camarilla S3 = support level, break below = short
        long_entry = (close_val > camarilla_r3_val) and bullish_1w and vol_spike
        short_entry = (close_val < camarilla_s3_val) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Camarilla level touch (or trend reversal)
        exit_long = (close_val < camarilla_s3_val) or not bullish_1w
        exit_short = (close_val > camarilla_r3_val) or not bearish_1w
        
        # Minimum holding period: 3 bars (~12h for 4h timeframe)
        min_hold = 3
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0