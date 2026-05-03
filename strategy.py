#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets: buy when Bear Power improves (less negative) + price above 1d EMA50 + volume spike
# In bear markets: sell when Bull Power deteriorates (less positive) + price below 1d EMA50 + volume spike
# Works in both regimes by measuring underlying momentum strength/weakness
# Volume spike (>1.8x 20-period EMA) confirms institutional participation
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Elder Ray signals with 1d trend filter
        # Long: Bear Power improving (less negative) + price above 1d EMA50 + volume spike
        # Short: Bull Power deteriorating (less positive) + price below 1d EMA50 + volume spike
        if position == 0:
            if (bear_power[i] > bear_power[i-1] and bear_power[i] < 0 and  # Improving but still negative
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (bull_power[i] < bull_power[i-1] and bull_power[i] > 0 and  # Deteriorating but still positive
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power deteriorates (more negative) OR price below 1d EMA50
            if bear_power[i] < bear_power[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power improves (more positive) OR price above 1d EMA50
            if bull_power[i] > bull_power[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals