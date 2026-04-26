#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2
Hypothesis: Trade breakouts of Camarilla R3/S3 levels from 1d, filtered by 1d EMA34 trend and volume spike.
Camarilla levels provide intraday support/resistance; breaks indicate strong momentum.
1d EMA34 filters for trend alignment, volume spike confirms institutional interest.
Designed for low trade frequency (20-50/year) to minimize fee drag on 4h timeframe.
Works in both bull and bear markets by following 1d trend and requiring volume confirmation.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high - low), S3 = close - 1.1*(high - low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 2.0x median volume on 1d
    vol_median_1d = pd.Series(df_1d['volume']).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (34), volume median (30)
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        
        if position == 0:
            # Long: close breaks above R3, uptrend (close > EMA34), volume spike
            long_signal = (close_val > r3_level) and \
                          (close_val > ema_34_1d_val) and \
                          (volume_val > 2.0 * vol_median_1d_val)
            # Short: close breaks below S3, downtrend (close < EMA34), volume spike
            short_signal = (close_val < s3_level) and \
                           (close_val < ema_34_1d_val) and \
                           (volume_val > 2.0 * vol_median_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: trend reversal (close < EMA34) or price retracement to mid-level after minimum holding
            mid_level = (r3_level + s3_level) / 2
            if bars_since_entry >= 6 and ((close_val < ema_34_1d_val) or (close_val < mid_level)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: trend reversal (close > EMA34) or price retracement to mid-level after minimum holding
            mid_level = (r3_level + s3_level) / 2
            if bars_since_entry >= 6 and ((close_val > ema_34_1d_val) or (close_val > mid_level)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0