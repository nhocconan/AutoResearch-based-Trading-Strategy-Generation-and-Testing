#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses 1w Camarilla pivot levels (R3/S3) for breakout entries with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 AND 1d close > EMA34 (uptrend) AND volume > 2.0 * 20-period average.
Short when price breaks below S3 AND 1d close < EMA34 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to the pivot level (R3 for longs, S3 for shorts) OR trend reverses.
Designed for 12h timeframe to achieve 50-150 total trades over 4 years with low fee drag.
Uses 1w HTF for stronger trend filter to avoid false breakouts in choppy markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for Camarilla pivots (R3, S3 levels)
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w Camarilla pivot levels: R3, S3
    # Camarilla formulas: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 with 1d trend filter AND volume
            # Long: price breaks above R3 (strong resistance) AND 1d uptrend AND volume
            long_condition = (close_val > r3_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S3 (strong support) AND 1d downtrend AND volume
            short_condition = (close_val < s3_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R3 level OR trend breaks
            exit_condition = (close_val <= r3_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S3 level OR trend breaks
            exit_condition = (close_val >= s3_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0