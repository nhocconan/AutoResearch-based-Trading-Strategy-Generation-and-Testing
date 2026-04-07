#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray with 1d regime filter and volume confirmation
# Hypothesis: Elder Ray power (bull/bear) captures institutional pressure; 1d trend filter avoids counter-trend trades; volume confirms conviction.
# Works in bull via bull power > 0 + uptrend, in bear via bear power < 0 + downtrend. Volume filter avoids low-conviction whipsaws.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
name = "6h_elder_ray_1d_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (using close)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d trend (50-period EMA slope) for regime filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    # Trend: 1 if rising (current > previous), -1 if falling, 0 if flat
    trend_1d = np.zeros_like(ema50_1d_aligned)
    trend_1d[1:] = np.where(ema50_1d_aligned[1:] > ema50_1d_aligned[:-1], 1,
                            np.where(ema50_1d_aligned[1:] < ema50_1d_aligned[:-1], -1, 0))
    
    # Calculate 6h volume confirmation (volume > 20-period average)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 20-period average
        vol_confirm = volume[i] > vol_ma_6h[i]
        
        if position == 1:  # Long position
            # Exit: bear power becomes positive (selling pressure) OR trend turns down OR volume fails
            if bear_power_1d_aligned[i] > 0 or trend_1d[i] == -1 or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: bull power becomes negative (buying pressure) OR trend turns up OR volume fails
            if bull_power_1d_aligned[i] < 0 or trend_1d[i] == 1 or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: bull power positive (buying pressure) + uptrend + volume confirmation
            if bull_power_1d_aligned[i] > 0 and trend_1d[i] == 1 and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: bear power negative (selling pressure) + downtrend + volume confirmation
            elif bear_power_1d_aligned[i] < 0 and trend_1d[i] == -1 and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals