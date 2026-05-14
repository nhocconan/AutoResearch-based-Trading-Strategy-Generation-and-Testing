#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn
Hypothesis: Camarilla R3/S3 breakouts on 12h timeframe with 1d EMA34 trend filter and dynamic volume spike confirmation. 
The strategy trades in the direction of the daily trend (EMA34) to catch momentum moves while avoiding counter-trend trades.
Volume spike (>2.0x 20-period MA) confirms institutional participation. Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the daily trend direction, which adapts to regime changes.
Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA + 20 for volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: close breaks above R3, with 1d uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and uptrend_1d[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3, with 1d downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and downtrend_1d[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below S3 (mean reversion) OR 1d trend changes to downtrend
            if (close[i] < camarilla_s3_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above R3 (mean reversion) OR 1d trend changes to uptrend
            if (close[i] > camarilla_r3_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn"
timeframe = "12h"
leverage = 1.0