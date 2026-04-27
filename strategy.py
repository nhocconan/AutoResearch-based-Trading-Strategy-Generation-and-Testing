#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h strategy using Camarilla R3/S3 levels from 1w for breakout entries with 1w EMA34 trend filter and volume confirmation.
Enter long when price closes above R3 with 1w uptrend (price > EMA34) and volume > 2.0x 20-period average.
Enter short when price closes below S3 with 1w downtrend (price < EMA34) and volume confirmation.
Exit on opposite Camarilla level touch (S3/R3) or 1w trend reversal (price crosses EMA34).
Designed for low trade frequency (~12-37/year) with discrete position sizing (0.25) to minimize fee drag and improve test generalization.
Works in both bull and bear markets by following the 1w trend while using Camarilla levels for precise breakout entries.
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
    
    # Get 1w data for Camarilla levels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w OHLC for Camarilla levels
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R3/S3 from 1w OHLC (wider than R1/S1 for fewer false breakouts)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1w + (h_1w - l_1w) * 1.1 / 4
    camarilla_s3 = c_1w - (h_1w - l_1w) * 1.1 / 4
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(c_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 12h timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1w EMA34 (34) + volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike
            # Long: price closes above R3 AND above EMA34 (1w uptrend)
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S3 AND below EMA34 (1w downtrend)
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level) OR 1w EMA34 turns bearish (price below EMA)
            if (close_val < s3_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level) OR 1w EMA34 turns bullish (price above EMA)
            if (close_val > r3_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0