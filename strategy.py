#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 for strong trend alignment (avoids whipsaws in ranging markets)
# Donchian(20) from prior 12h session provides clear breakout levels
# Volume confirmation (>2.0x 50 EMA) filters low-participation false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn. Target: 60-120 trades over 4 years.

name = "12h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (completed weekly bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get prior 12h data for Donchian levels (based on completed session)
    df_12h_prior = get_htf_data(prices, '12h')
    if len(df_12h_prior) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) from completed 12h session
    high_12h = df_12h_prior['high'].values
    low_12h = df_12h_prior['low'].values
    
    # Donchian upper(20) and lower(20) from prior completed 12h bars
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (completed 12h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h_prior, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h_prior, donchian_low)
    
    # Volume confirmation: 50-period EMA of volume on 12h timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 50-period EMA (strict filter)
        volume_confirm = volume[i] > (2.0 * vol_ema_50[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian High + uptrend + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian Low + downtrend + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals