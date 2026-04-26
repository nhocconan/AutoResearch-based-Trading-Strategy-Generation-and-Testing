#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1
Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA34 trend filter and volume spike confirmation.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Long when price breaks above Camarilla R3 level AND weekly uptrend AND volume spike
- Short when price breaks below Camarilla S3 level AND weekly downtrend AND volume spike
- Camarilla levels from prior day provide strong intraday support/resistance
- Weekly EMA34 trend filter reduces whipsaw and captures major multi-week moves
- Volume spike (2.0x 20-period average) confirms institutional participation
- Designed for lower frequency with proven edge on BTC/ETH from Camarilla's structure
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
    
    # Load prior 1d data ONCE before loop for Camarilla levels (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    
    # Load weekly data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    prior_1d_high = np.roll(df_1d['high'].values, 1)
    prior_1d_low = np.roll(df_1d['low'].values, 1)
    prior_1d_close = np.roll(df_1d['close'].values, 1)
    # First value is invalid due to roll
    prior_1d_high[0] = np.nan
    prior_1d_low[0] = np.nan
    prior_1d_close[0] = np.nan
    
    cam_r3 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    cam_s3 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (no alignment needed as both are daily)
    cam_r3_aligned = cam_r3  # Already aligned to daily bars
    cam_s3_aligned = cam_s3  # Already aligned to daily bars
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume spike (20-period volume average on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior day, 34 for weekly EMA)
    start_idx = max(20, 1, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND weekly uptrend AND volume spike
            if close[i] > cam_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND weekly downtrend AND volume spike
            elif close[i] < cam_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR weekly trend turns down
            if close[i] < cam_s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR weekly trend turns up
            if close[i] > cam_r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0