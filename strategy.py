#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d Williams %R regime + volume confirmation
# In trending markets (1d Williams %R between -80 and -20), we trade breakouts in trend direction: long on upper breakout, short on lower breakout.
# In ranging markets (1d Williams %R above -20 or below -80), we fade extremes: short near upper band, long near lower band.
# Volume confirmation (>1.5x 20-period EMA) reduces false breakouts. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_1dWilliamsR_Regime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    close_1d = pd.Series(df_1d['close'])
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max()
    lowest_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high_20.values
    donchian_lower = lowest_low_20.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine regime: ranging (Williams %R > -20 or < -80) or trending (between -80 and -20)
            if williams_r_aligned[i] > -20 or williams_r_aligned[i] < -80:
                # Ranging market: fade extremes (mean reversion)
                if close[i] <= donchian_lower_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_upper_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Trending market: trade breakouts in trend direction
                # Long: upper breakout
                if (close[i] > donchian_upper_aligned[i] and 
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                # Short: lower breakout
                elif (close[i] < donchian_lower_aligned[i] and 
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint OR Williams %R enters overbought (> -20) OR volume drops
            if (close[i] <= donchian_mid_aligned[i] or 
                williams_r_aligned[i] > -20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint OR Williams %R enters oversold (< -80) OR volume drops
            if (close[i] >= donchian_mid_aligned[i] or 
                williams_r_aligned[i] < -80 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals