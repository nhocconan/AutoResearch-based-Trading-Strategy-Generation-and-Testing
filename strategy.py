#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: 6h Camarilla R3/S3 breakout with volume spike and 1w EMA34 trend filter.
- Long when price breaks above Camarilla R3 level (from prior 1d range) AND volume spike AND 1w EMA34 uptrend
- Short when price breaks below Camarilla S3 level AND volume spike AND 1w EMA34 downtrend
- Uses prior 1d range for Camarilla levels (structure from daily session)
- Volume spike confirms institutional participation (2.5x 24-period average on 6h to reduce noise)
- 1w EMA34 filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws in bear markets)
- Designed for lower frequency (target 12-37 trades/year on 6h) to minimize fee drag and improve test generalization
- Exit on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Weekly EMA34 trend filter on 6h timeframe with Camarilla R3/S3 (wider bands for stronger breaks)
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
    
    # Load 1d data ONCE before loop for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (completed bar only)
    # Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    prior_1d_high = np.roll(df_1d['high'].values, 1)
    prior_1d_low = np.roll(df_1d['low'].values, 1)
    prior_1d_close = np.roll(df_1d['close'].values, 1)
    # First value is invalid due to roll
    prior_1d_high[0] = np.nan
    prior_1d_low[0] = np.nan
    prior_1d_close[0] = np.nan
    
    cam_r3 = prior_1d_close + (prior_1d_high - prior_1d_low) * 1.1 / 4
    cam_s3 = prior_1d_close - (prior_1d_high - prior_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed for structure)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Load 1w data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter (needs completed 1w candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                         np.where(close > ema_34_1w_aligned, 1, -1), 
                         0)
    
    # Calculate volume spike (24-period volume average on 6h - more stringent)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma24 * 2.5)  # Volume at least 2.5x average to reduce false signals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 24 for volume MA, 1 for prior 1d, 34 for 1w EMA)
    start_idx = max(24, 1, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and 1w trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND volume spike AND 1w uptrend
            if close[i] > cam_r3_aligned[i] and volume_spike[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND volume spike AND 1w downtrend
            elif close[i] < cam_s3_aligned[i] and volume_spike[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1w trend turns down
            if close[i] < cam_s3_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1w trend turns up
            if close[i] > cam_r3_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0