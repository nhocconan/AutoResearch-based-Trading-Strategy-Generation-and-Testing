#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR regime filter
# - Primary signal: Price breaks above/below 12h Donchian channel (20-period high/low)
# - Trend filter: 1d ATR(14) > 20-period median ATR (avoid low-volatility chop)
# - Volume confirmation: 12h volume > 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: ATR-based trailing stop (signal=0 when price < highest - 2.5*ATR for longs,
#   price > lowest + 2.5*ATR for shorts) using only close prices
# - Works in bull/bear: Donchian breakouts capture trends, volume/ATR filter ensures
#   trades occur during energetic moves, reducing whipsaws in ranging markets

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility regime
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 
                                   np.abs(high_1d[0] - close_1d[0]),
                                   np.abs(low_1d[0] - close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 12h timeframe (completed 1d bar only)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track highest high since entry for trailing stop (longs)
    # Track lowest low since entry for trailing stop (shorts)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_14_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if np.isnan(highest_since_entry[i-1]):
                highest_since_entry[i] = high[i]
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # Exit: Price closes below Donchian low OR ATR trailing stop hit
            donchian_exit = close[i] < lowest_low_20[i]
            atr_stop = close[i] < highest_since_entry[i] - 2.5 * atr_14_aligned[i]
            
            if donchian_exit or atr_stop:
                position = 0
                signals[i] = 0.0
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if np.isnan(lowest_since_entry[i-1]):
                lowest_since_entry[i] = low[i]
            else:
                lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # Exit: Price closes above Donchian high OR ATR trailing stop hit
            donchian_exit = close[i] > highest_high_20[i]
            atr_stop = close[i] > lowest_since_entry[i] + 2.5 * atr_14_aligned[i]
            
            if donchian_exit or atr_stop:
                position = 0
                signals[i] = 0.0
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and ATR regime filter
            # Long: Price closes above Donchian high AND volume regime AND ATR > median
            if (close[i] > highest_high_20[i] and 
                volume_regime[i] and 
                atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])):
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize tracking
            # Short: Price closes below Donchian low AND volume regime AND ATR > median
            elif (close[i] < lowest_low_20[i] and 
                  volume_regime[i] and 
                  atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])):
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize tracking
    
    return signals