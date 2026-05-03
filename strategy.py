#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; signals when power expands with trend alignment.
# Volume confirmation (1.5x 20-period EMA) filters low-momentum false signals.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "6h_ElderRay_EMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h close)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start from 34 to have valid EMA13 and Elder Ray
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power expanding (current > previous) + above 1d EMA34 + volume spike
            if bull_power[i] > bull_power[i-1] and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power contracting (current < previous) + below 1d EMA34 + volume spike
            elif bear_power[i] < bear_power[i-1] and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power contracting or loses 1d trend alignment
            if bull_power[i] < bull_power[i-1] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power expanding or loses 1d trend alignment
            if bear_power[i] > bear_power[i-1] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals