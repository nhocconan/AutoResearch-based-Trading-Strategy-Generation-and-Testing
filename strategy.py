#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA(34) trend filter and volume confirmation
# Elder Ray measures bull/bear strength: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, price above 1d EMA34, and volume spike
# Short when Bear Power < 0 and falling, price below 1d EMA34, and volume spike
# Works in bull/bear markets by following 1d trend and requiring volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    # Handle first 3 values
    bull_power_slope[:3] = 0
    bear_power_slope[:3] = 0
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMAs and volume EMA
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising + price above 1d EMA34 + volume spike
            if bull_power[i] > 0 and bull_power_slope[i] > 0 and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling + price below 1d EMA34 + volume spike
            elif bear_power[i] < 0 and bear_power_slope[i] < 0 and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or loses 1d trend alignment
            if bull_power[i] <= 0 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or loses 1d trend alignment
            if bear_power[i] >= 0 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals