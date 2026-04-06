#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly trend filter + volume confirmation
# Enter long when: price breaks above Donchian upper band, weekly EMA(21) rising, volume > 1.5x avg
# Enter short when: price breaks below Donchian lower band, weekly EMA(21) falling, volume > 1.5x avg
# Exit when price returns to Donchian median or opposite breakout occurs
# Targets 50-100 trades over 4 years with strong trend capture in both bull/bear markets

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
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
    
    # Donchian(20) on daily
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian median OR opposite breakout
            if close[i] <= donchian_mid[i] or low[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian median OR opposite breakout
            if close[i] >= donchian_mid[i] or high[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            if volume[i] > volume_threshold[i]:
                # Bullish breakout: price above upper band with rising weekly EMA
                if high[i] > high_max[i] and ema_21_aligned[i] > ema_21_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower band with falling weekly EMA
                elif low[i] < low_min[i] and ema_21_aligned[i] < ema_21_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
    
    return signals