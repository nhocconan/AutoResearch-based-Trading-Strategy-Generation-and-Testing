#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses 1w Camarilla pivot levels (R3/S3) from weekly data for breakout signals. In uptrend (price > weekly EMA34), go long when price breaks above R3 with volume confirmation; in downtrend (price < weekly EMA34), go short when price breaks below S3 with volume confirmation. Exit when price reverts to the weekly EMA34 or opposite Camarilla level is touched. Weekly timeframe provides strong trend filter for 12h entries, minimizing whipsaw in ranging markets. Volume spike (>2x average) ensures institutional participation. Designed for 50-150 trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for Camarilla
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(weekly_close)
    weekly_ema34 = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels (R3, S3, R4, S4) from previous weekly bar
    # Camarilla formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    #                  S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    weekly_range = weekly_high - weekly_low
    camarilla_R3 = weekly_close + (weekly_range * 1.1 / 4)
    camarilla_S3 = weekly_close - (weekly_range * 1.1 / 4)
    camarilla_R4 = weekly_close + (weekly_range * 1.1 / 2)
    camarilla_S4 = weekly_close - (weekly_range * 1.1 / 2)
    
    # Align all 1w indicators to 12h timeframe (wait for weekly bar to close)
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly EMA34 (34) and volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_ema34_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_ema = weekly_ema34_aligned[i]
        r3_level = camarilla_R3_aligned[i]
        s3_level = camarilla_S3_aligned[i]
        r4_level = camarilla_R4_aligned[i]
        s4_level = camarilla_S4_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > weekly EMA34 = uptrend, price < weekly EMA34 = downtrend
            is_uptrend = close_val > weekly_ema
            is_downtrend = close_val < weekly_ema
            
            if is_uptrend:
                # Uptrend: long when price breaks above R3 with volume confirmation
                if (close_val > r3_level) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S3 with volume confirmation
                if (close_val < s3_level) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price reverts to weekly EMA or touches S4 (strong reversal)
            exit_condition = (close_val < weekly_ema) or (close_val < s4_level)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reverts to weekly EMA or touches R4 (strong reversal)
            exit_condition = (close_val > weekly_ema) or (close_val > r4_level)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0