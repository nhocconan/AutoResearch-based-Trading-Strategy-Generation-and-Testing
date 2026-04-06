#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Supertrend + volume confirmation
# Enter long when: Williams %R(14) < -80, Supertrend(1d) = bullish, volume > 1.5x average
# Enter short when: Williams %R(14) > -20, Supertrend(1d) = bearish, volume > 1.5x average
# Exit when: Williams %R crosses above -50 (long) or below -50 (short) OR opposite Supertrend
# Target: 80-180 trades over 4 years by combining extreme momentum with trend filter
# Williams %R identifies overbought/oversold, Supertrend filters counter-trend moves

name = "4h_williams_r_1d_supertrend_vol_v1"
timeframe = "4h"
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
    
    # Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    williams_r = williams_r.values
    
    # Supertrend on 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for Supertrend
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    
    # Supertrend parameters
    atr_mult = 3.0
    upper_band = (high_1d + low_1d) / 2 + atr_mult * atr.values
    lower_band = (high_1d + low_1d) / 2 - atr_mult * atr.values
    
    # Initialize Supertrend
    supertrend = np.full(len(close_1d), np.nan)
    direction = np.full(len(close_1d), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if np.isnan(supertrend[i-1]):
            # First valid value
            if close_1d[i] > upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1  # downtrend
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1   # uptrend
        else:
            if direction[i-1] == 1:  # was uptrend
                if close_1d[i] <= supertrend[i-1]:
                    # trend change to downtrend
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    # remain uptrend
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                    direction[i] = 1
            else:  # was downtrend
                if close_1d[i] >= supertrend[i-1]:
                    # trend change to uptrend
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    # remain downtrend
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
                    direction[i] = -1
    
    # Align Supertrend direction to 4h
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Williams %R > -50 OR Supertrend turns bearish
            if williams_r[i] > -50 or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -50 OR Supertrend turns bullish
            if williams_r[i] < -50 or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Extreme Williams %R + Supertrend alignment + volume
            if volume[i] > volume_threshold[i]:
                if williams_r[i] < -80 and direction_aligned[i] == 1:
                    # Oversold with bullish Supertrend
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20 and direction_aligned[i] == -1:
                    # Overbought with bearish Supertrend
                    signals[i] = -0.25
                    position = -1
    
    return signals