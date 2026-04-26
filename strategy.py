#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: 6h Camarilla R3/S3 breakout with volume spike and 12h EMA50 trend filter.
- Long when price breaks above Camarilla R3 level (from prior 6h range) AND volume spike AND 12h EMA50 uptrend
- Short when price breaks below Camarilla S3 level AND volume spike AND 12h EMA50 downtrend
- Uses prior 6h range for Camarilla levels (structure-based edge from prior completed 6h bar)
- Volume spike confirms institutional participation (2.0x 20-period average on 6h)
- 12h EMA50 filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
- Designed for lower frequency (target 12-37 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Camarilla R3/S3 levels (stronger support/resistance than R1/S1) + 12h HTF trend filter on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Camarilla levels (structure)
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Camarilla levels from prior 6h bar (completed bar only)
    # Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    prior_6h_high = np.roll(df_6h['high'].values, 1)
    prior_6h_low = np.roll(df_6h['low'].values, 1)
    prior_6h_close = np.roll(df_6h['close'].values, 1)
    # First value is invalid due to roll
    prior_6h_high[0] = np.nan
    prior_6h_low[0] = np.nan
    prior_6h_close[0] = np.nan
    
    cam_r3 = prior_6h_close + (prior_6h_high - prior_6h_low) * 1.1 / 4
    cam_s3 = prior_6h_close - (prior_6h_high - prior_6h_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed for structure)
    cam_r3_aligned = align_htf_to_ltf(prices, df_6h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_6h, cam_s3)
    
    # Load 12h data ONCE before loop for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter (needs completed 12h candle)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_12h = np.where(ema_50_12h_aligned > 0, 
                         np.where(close > ema_50_12h_aligned, 1, -1), 
                         0)
    
    # Calculate volume spike (20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior 6h, 50 for 12h EMA)
    start_idx = max(20, 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_12h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and 12h trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND volume spike AND 12h uptrend
            if close[i] > cam_r3_aligned[i] and volume_spike[i] and trend_12h[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND volume spike AND 12h downtrend
            elif close[i] < cam_s3_aligned[i] and volume_spike[i] and trend_12h[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 12h trend turns down
            if close[i] < cam_s3_aligned[i] or trend_12h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 12h trend turns up
            if close[i] > cam_r3_aligned[i] or trend_12h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0