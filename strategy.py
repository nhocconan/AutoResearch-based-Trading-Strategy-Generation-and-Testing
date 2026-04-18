#!/usr/bin/env python3
"""
6h Elder Ray Index with 1d Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
bull/bear strength relative to trend. In strong trends (EMA50 slope), we take
trend-aligned signals: buy when Bull Power turns positive in uptrend, sell when
Bear Power turns positive in downtrend. Volume confirms institutional participation.
Works in bull markets via uptrend longs and in bear markets via downtrend shorts.
Target: 20-50 trades/year on 6h timeframe.
"""

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
    
    # Elder Ray components on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d trend filter: EMA50 slope
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    # Slope: current - 3 periods ago (to avoid noise)
    ema50_slope = ema50_1d_aligned - np.roll(ema50_1d_aligned, 3)
    ema50_slope[:3] = 0  # first 3 values invalid
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(ema50_slope[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power turns positive in uptrend with volume
            if bull_power[i] > 0 and bull_power[i-1] <= 0 and ema50_slope[i] > 0 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power turns positive in downtrend with volume
            elif bear_power[i] > 0 and bear_power[i-1] <= 0 and ema50_slope[i] < 0 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR trend turns down
            if bull_power[i] <= 0 or ema50_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: Bear Power turns negative OR trend turns up
            if bear_power[i] <= 0 or ema50_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_1dTrendFilter_Volume"
timeframe = "6h"
leverage = 1.0
EOF