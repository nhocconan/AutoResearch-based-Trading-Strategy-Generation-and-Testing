#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 filter and volume spike confirmation.
# Elder Ray measures bull/bear power as price deviation from EMA, indicating trend strength.
# 12h EMA34 filter ensures we trade only in the direction of the higher timeframe trend.
# Volume spike confirms conviction in the move.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (bull power positive with uptrend) and bear markets (bear power negative with downtrend).
name = "6h_ElderRay_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 13-period EMA as per standard Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema34_12h_aligned[i]
        downtrend = close[i] < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: bull power positive AND uptrend AND volume spike
            if bull_power[i] > 0 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative AND downtrend AND volume spike
            elif bear_power[i] < 0 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative OR trend reverses
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive OR trend reverses
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals