#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when price breaks above R3 AND 1w uptrend AND volume spike
- Short when price breaks below S3 AND 1w downtrend AND volume spike
- Camarilla levels provide structure; 1w EMA34 filters bear market whipsaw
- Volume spike (2.0x 20-period average) confirms institutional participation
- Designed for low frequency with proven edge on BTC/ETH from Camarilla's accuracy in ranging/breakout markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d EMA34 for additional trend confirmation (optional but helps)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike (20-period volume average on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla levels: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # We use 1d data to compute levels for current 12h bar
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    close_1d = df_1d_for_camarilla['close'].values
    high_1d = df_1d_for_camarilla['high'].values
    low_1d = df_1d_for_camarilla['low'].values
    
    # Camarilla width
    camarilla_width = 1.1 * (high_1d - low_1d)
    
    # R3 and S3 levels (most significant for breakout)
    r3 = close_1d + camarilla_width * 1.1 / 4
    s3 = close_1d - camarilla_width * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
            # Long: Price breaks above R3 AND 1w uptrend AND 1d uptrend AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1w_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND 1w downtrend AND 1d downtrend AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1w_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below S3 OR 1w trend turns down OR 1d trend turns down
            if close[i] < s3_aligned[i] or close[i] < ema34_1w_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above R3 OR 1w trend turns up OR 1d trend turns up
            if close[i] > r3_aligned[i] or close[i] > ema34_1w_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0