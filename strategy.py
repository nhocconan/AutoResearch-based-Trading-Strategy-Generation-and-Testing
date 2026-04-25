#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts in direction of 1d trend (EMA34) with volume confirmation.
Camarilla levels provide high-probability reversal/breakout points. Trend filter avoids counter-trend trades.
Volume spike confirms institutional participation. Designed for low trade frequency (12-37/year) to minimize fee drag.
Works in bull markets (breakouts with trend) and bear markets (breakouts with trend, short bias).
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
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average (1d) for volume spike filter
    vol_avg_20_1d = pd.Series(df_1d['volume'].values if 'volume' in df_1d.columns else close_1d * 0).rolling(
        window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla R1, R3, S1, S3 levels
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    camarilla_r3 = typical_price_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = typical_price_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and volume average (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * 20-day average volume
        volume_spike = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_34_1d_aligned[i] and \
                         volume_spike
            # Short: price breaks below S1 AND 1d trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close_1d_align := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_34_1d_aligned[i] and \
                          volume_spike
            
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
            # Exit: price re-enters between S1 and R1 OR 1d trend turns bearish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between S1 and R1 OR 1d trend turns bullish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (align_htf_to_ltf(prices, df_1d, close_1d)[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0