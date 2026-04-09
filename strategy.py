#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with volume confirmation
# ATR(14) expansion on 1d indicates regime shift, traded on 12h breakouts of prior 12h high/low
# Volume confirmation (current 12h volume > 1.5x 20-period average) filters false signals
# Works in bull/bear: volatility expansion precedes sustained moves in both directions
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_atr_breakout_volume_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # ATR expansion signal: current ATR > 1.2x 20-period ATR average
    atr_ma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_14_1d > 1.2 * atr_ma_20_1d
    
    # Align ATR expansion to 12h timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # Calculate 12h rolling high/low for breakout levels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_expansion_aligned[i]) or
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to 12h VWAP approximation (midpoint of range)
            exit_level = (high_ma_20[i] + low_ma_20[i]) / 2.0
            if close[i] < exit_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on retracement to 12h VWAP approximation
            exit_level = (high_ma_20[i] + low_ma_20[i]) / 2.0
            if close[i] > exit_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volatility and volume confirmation
            # Long on break above 12h 20-period high, Short on break below 12h 20-period low
            if atr_expansion_aligned[i] and volume_confirmed:
                if close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals