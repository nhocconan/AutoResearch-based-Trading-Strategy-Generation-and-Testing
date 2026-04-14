#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour strategy using 1-day Williams %R for mean reversion and 1-day ATR for volatility filtering.
# In oversold conditions (Williams %R < -80) with low volatility (ATR below median), go long.
# In overbought conditions (Williams %R > -20) with low volatility, go short.
# Uses 1-day timeframe for stable signals less prone to whipsaw in both bull and bear markets.
# Volume confirmation: current volume > 1.5x 20-period average to ensure participation.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for mean reversion and volatility filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14-period) on 1d
    williams_len = 14
    if len(df_1d) < williams_len:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high and lowest low over williams_len period
    highest_high = pd.Series(high_1d).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low_1d).rolling(window=williams_len, min_periods=williams_len).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR (14-period) on 1d for volatility filter
    atr_len = 14
    if len(df_1d) < atr_len:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Calculate ATR median for volatility threshold
    atr_median = np.nanmedian(atr_1d[~np.isnan(atr_1d)])
    
    # Williams %R and ATR alignment to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(williams_len*2, atr_len*2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is below median (low volatility regime)
        low_volatility = atr_1d_aligned[i] < atr_median
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if low_volatility and volume_confirmed:
                # Oversold condition: Williams %R < -80 -> long
                if williams_r_aligned[i] < -80:
                    position = 1
                    signals[i] = position_size
                # Overbought condition: Williams %R > -20 -> short
                elif williams_r_aligned[i] > -20:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to oversold threshold or becomes overbought
            if williams_r_aligned[i] > -20:  # Reached overbought territory
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to overbought threshold or becomes oversold
            if williams_r_aligned[i] < -80:  # Reached oversold territory
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_VolatilityFilter_Volume"
timeframe = "4h"
leverage = 1.0