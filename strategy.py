#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
- Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
- Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
- Uses prior 1d range for Camarilla levels (structure-based edge)
- Volume spike confirms institutional participation (2.0x 12-period average)
- 1d EMA34 trend filter ensures alignment with higher timeframe trend
- Designed for low frequency (target 12-37 trades/year) with proven edge on BTC/ETH
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
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # Align Camarilla levels to 12h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike (12-period volume average on 12h)
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (vol_ma12 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 12 for volume MA, 1 for prior day, 34 for EMA)
    start_idx = max(12, 34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and 1d trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if close[i] > cam_r3_aligned[i] and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif close[i] < cam_s3_aligned[i] and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3
            if close[i] < cam_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3
            if close[i] > cam_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0