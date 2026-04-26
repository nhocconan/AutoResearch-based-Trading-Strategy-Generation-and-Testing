#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirmation_v1
Hypothesis: 1d breakout above/below weekly Camarilla R3/S3 levels in direction of 1w EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Uses 1w HTF for trend alignment and Camarilla levels from weekly OHLC for institutional support/resistance. Volume confirmation reduces false breakouts. Designed for 7-25 trades/year (30-100 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1w trend while using Camarilla structure for precise entries.
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
    
    # Calculate Camarilla levels from previous 1w bar (OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3, S3 levels (based on previous week's range)
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed as they're based on completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema)
    start_idx = max(20, 34)
    
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
        
        # Entry conditions: breakout of Camarilla level in trend direction with volume
        long_entry = (close_val > camarilla_r3_val) and bullish_1w and vol_spike
        short_entry = (close_val < camarilla_s3_val) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Camarilla level touch (or trend reversal)
        exit_long = (close_val < camarilla_s3_val) or not bullish_1w
        exit_short = (close_val > camarilla_r3_val) or not bearish_1w
        
        # Minimum holding period: 3 bars
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

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0