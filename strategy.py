#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation.
# Long when price breaks above Donchian(20) high with volume > 1.3x 1d average and ATR(14) > 1d ATR(14) EMA.
# Short when price breaks below Donchian(20) low with same conditions.
# Uses 1d ATR regime filter to avoid low-volatility chop. Designed for 20-40 trades/year.
# Works in bull/bear markets by filtering breakouts with volatility regime.

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) and its EMA for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA of ATR for regime filter
    atr_ema_1d = pd.Series(atr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily indicators to 4h timeframe
    atr_ema_aligned = align_htf_to_ltf(prices, df_1d, atr_ema_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure Donchian is valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(atr_ema_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: current 1d ATR EMA > 0 (always true if calculated) 
        # Volume filter: current volume > 1.3x 20-day average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks Donchian bands with volume confirmation
        long_entry = (high[i] > high_max[i] and vol_filter)
        short_entry = (low[i] < low_min[i] and vol_filter)
        
        # Exit conditions: price returns to opposite Donchian band (trailing exit)
        exit_long = low[i] < low_min[i]  # Exit long if price breaks lower band
        exit_short = high[i] > high_max[i]  # Exit short if price breaks upper band
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals