#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Only take longs when price breaks above R1 in 1d uptrend with volume > 1.5x 20-period average. 
Only take shorts when price breaks below S1 in 1d downtrend with volume confirmation. 
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year.
Works in bull/bear via trend alignment: follows 1d trend direction only.
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
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We need previous day's OHLC for today's levels
    # Since we're on 4h timeframe, we'll use 1d data shifted by 1
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long signal: price breaks above R1 in uptrend with volume spike
        if close[i] > R1_4h[i] and htf_trend[i] == 1 and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short signal: price breaks below S1 in downtrend with volume spike
        elif close[i] < S1_4h[i] and htf_trend[i] == -1 and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to midpoint or opposite level
        elif position == 1 and (close[i] < (R1_4h[i] + S1_4h[i])/2 or close[i] < S1_4h[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (R1_4h[i] + S1_4h[i])/2 or close[i] > R1_4h[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0