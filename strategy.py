#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike (>2x 20-bar avg volume) and 1d EMA50 trend filter. 
# Uses discrete sizing 0.25 to target 75-150 total trades over 4 years on 4h timeframe.
# Camarilla levels provide institutional support/resistance; volume spike confirms participation; 
# 1d EMA50 ensures higher timeframe trend alignment. Designed for fewer, higher-quality trades 
# to minimize fee drag while working in both bull and bear markets by capturing strong breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume_1d = pd.Series(df_1d['volume'].values).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate Camarilla levels from prior 1d candle only
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    prior_close_1d = df_1d['close'].shift(1).values
    prior_high_1d = df_1d['high'].shift(1).values
    prior_low_1d = df_1d['low'].shift(1).values
    prior_range_1d = prior_high_1d - prior_low_1d
    camarilla_r3 = prior_close_1d + 1.1 * prior_range_1d * 1.1 / 4
    camarilla_s3 = prior_close_1d - 1.1 * prior_range_1d * 1.1 / 4
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for all indicators
    start_idx = max(50, lookback_vol)  # EMA50 and volume lookback
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA50, volume spike (>2x avg)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA50, volume spike (>2x avg)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3_aligned[i] or 
                volume[i] < avg_volume_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals