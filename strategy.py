#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn
Hypothesis: Trade 4h Camarilla R3/S3 level breaks in direction of 1d EMA34 trend with volume spike confirmation.
Camarilla levels provide high-probability intraday support/resistance. 1d EMA34 filter ensures trend alignment.
Volume spike confirms institutional participation. Designed for low trade frequency (20-50/year) to minimize fee drag.
Works in bull/bear via trend filter + volume confirmation to avoid whipsaws.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H+L+C)/3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_1d_vals = typical_price_1d.values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla R3, S3 levels: R3 = PP + 1.1*(H-L), S3 = PP - 1.1*(H-L)
    camarilla_pp = typical_price_1d_vals
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + 1.1 * camarilla_range
    camarilla_s3 = camarilla_pp - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for volume MA (20) and 1d EMA34 (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 1d trend bullish AND volume spike
            long_setup = (close[i] > camarilla_r3_aligned[i]) and \
                         (close_1d := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_34_1d_aligned[i] and \
                         volume_spike[i]
            # Short: price breaks below S3 AND 1d trend bearish AND volume spike
            short_setup = (close[i] < camarilla_s3_aligned[i]) and \
                          (close_1d := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_34_1d_aligned[i] and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla range (H3-L3) OR 1d trend turns bearish
            camarilla_h3 = camarilla_pp + 1.1 * camarilla_range / 4  # H3 = PP + 1.1*(H-L)/4
            camarilla_l3 = camarilla_pp - 1.1 * camarilla_range / 4  # L3 = PP - 1.1*(H-L)/4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if (close[i] < camarilla_h3_aligned[i] and close[i] > camarilla_l3_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla range (H3-L3) OR 1d trend turns bullish
            camarilla_h3 = camarilla_pp + 1.1 * camarilla_range / 4  # H3 = PP + 1.1*(H-L)/4
            camarilla_l3 = camarilla_pp - 1.1 * camarilla_range / 4  # L3 = PP - 1.1*(H-L)/4
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if (close[i] < camarilla_h3_aligned[i] and close[i] > camarilla_l3_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0