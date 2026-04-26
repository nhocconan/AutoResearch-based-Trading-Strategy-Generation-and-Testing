#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as intraday support/resistance on 4h chart.
Breakout above R1 with 1d uptrend (EMA34) and volume spike (>1.5x 20-period MA) triggers long.
Breakdown below S1 with 1d downtrend and volume spike triggers short.
Uses discrete position sizing (0.25) to minimize fee churn.
Camarilla levels calculated from prior 1d OHLC: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12.
Designed to work in both bull and bear markets by following the 1d trend.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + (1.1 * camarilla_range / 12)
    s1_level = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA + 20 for volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
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
            # Long: price > R1 with 1d uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with 1d downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < S1 (reversal below support) OR 1d trend changes to downtrend
            if (close[i] < s1_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > R1 (reversal above resistance) OR 1d trend changes to uptrend
            if (close[i] > r1_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0