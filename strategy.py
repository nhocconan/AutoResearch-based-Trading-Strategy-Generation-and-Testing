#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Long when price breaks above Camarilla R3 level AND 12h uptrend AND volume spike
- Short when price breaks below Camarilla S3 level AND 12h downtrend AND volume spike
- Camarilla levels provide intraday support/resistance based on prior day's range
- 12h EMA50 trend filter reduces whipsaw in bear markets and captures major moves
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
    
    # Load 1d data ONCE before loop for Camarilla levels (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use R3 and S3 as primary breakout levels
    prior_1d_high = np.roll(df_1d['high'].values, 1)
    prior_1d_low = np.roll(df_1d['low'].values, 1)
    prior_1d_close = np.roll(df_1d['close'].values, 1)
    # First value is invalid due to roll
    prior_1d_high[0] = np.nan
    prior_1d_low[0] = np.nan
    prior_1d_close[0] = np.nan
    
    cam_r3 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    cam_s3 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior day)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
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
            # Long: Price breaks above Camarilla R3 AND 12h uptrend AND volume spike
            if close[i] > cam_r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 12h downtrend AND volume spike
            elif close[i] < cam_s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 12h trend turns down
            if close[i] < cam_s3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 12h trend turns up
            if close[i] > cam_r3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0