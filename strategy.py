#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme with 1d EMA34 Trend Filter and Volume Spike.
- Williams %R(14) identifies overbought/oversold conditions; extreme readings (< -90 or > -10) signal high-probability reversals.
- 1d EMA34 provides higher-timeframe trend filter to align with intermediate momentum and reduce counter-trend trades.
- Volume spike (>2.0x 24-period average) confirms reversal validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 2.0x 24-period average volume (4h * 6 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -90 (extreme oversold) with volume spike and above 1d EMA34 (bullish higher-timeframe trend)
            if williams_r[i] < -90 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (extreme overbought) with volume spike and below 1d EMA34 (bearish higher-timeframe trend)
            elif williams_r[i] > -10 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (return from oversold) OR below 1d EMA34 (trend change)
            if williams_r[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (return from overbought) OR above 1d EMA34 (trend change)
            if williams_r[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0