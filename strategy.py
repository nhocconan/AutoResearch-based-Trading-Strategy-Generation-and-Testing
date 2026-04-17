#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Supertrend trend filter with price crossing above/below
the 12h Supertrend line as entry signal, confirmed by volume spike (1.5x 20-period volume MA)
and ATR-based volatility filter (ATR > 0.5 * 20-period ATR mean) to avoid low-volatility chop.
Exits when price crosses back below/above the Supertrend line.
Fixed position size 0.25 to manage drawdown. Designed for 4h timeframe with strict entry
conditions to limit trades to 75-200 total over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df_12h.index, dtype=float)
    direction = pd.Series(index=df_12h.index, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_12h)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df_12h['close'].iloc[i] <= upper_band.iloc[i-1]:
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])
            else:
                upper_band.iloc[i] = upper_band.iloc[i]
                
            if df_12h['close'].iloc[i] >= lower_band.iloc[i-1]:
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])
            else:
                lower_band.iloc[i] = lower_band.iloc[i]
            
            if direction.iloc[i-1] == 1:
                if df_12h['close'].iloc[i] <= lower_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
            else:
                if df_12h['close'].iloc[i] >= upper_band.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
    
    supertrend_values = supertrend.values
    direction_values = direction.values
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_values)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction_values)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # ATR-based volatility filter: ATR > 0.5 * 20-period ATR mean
    atr_14 = pd.Series(
        np.maximum(
            np.maximum(high - low, np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
    ).rolling(window=14, min_periods=14).mean()
    atr_mean_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean()
    volatility_filter = atr_14 > (0.5 * atr_mean_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 14, 30)  # warmup for volume MA, ATR, and Supertrend
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(atr_14.iloc[i]) or np.isnan(atr_mean_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        st_val = supertrend_aligned[i]
        dir_val = direction_aligned[i]
        vol_filter = volatility_filter.iloc[i]
        
        if position == 0:
            # Enter long when price crosses above Supertrend (uptrend) with volume spike and volatility
            if price > st_val and dir_val == 1 and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when price crosses below Supertrend (downtrend) with volume spike and volatility
            elif price < st_val and dir_val == -1 and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below Supertrend
            if price < st_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above Supertrend
            if price > st_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Supertrend12h_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0