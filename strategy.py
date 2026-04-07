#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Supertrend with Volume Confirmation
# Hypothesis: Supertrend (ATR-based trend filter) on daily timeframe provides
# robust trend direction that works in both bull and bear markets. Entry on 6s
# when price pulls back to Supertrend level with volume confirmation reduces
# whipsaw. Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_daily_supertrend_volume_v1"
timeframe = "6h"
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
    
    # Get daily data for Supertrend calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on daily timeframe
    # Parameters: ATR period = 10, multiplier = 3.0
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = df_daily['high'] - df_daily['low']
    tr2 = abs(df_daily['high'] - df_daily['close'].shift(1))
    tr3 = abs(df_daily['low'] - df_daily['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Calculate basic upper and lower bands
    hl_avg = (df_daily['high'] + df_daily['low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df_daily.index, dtype=float)
    direction = pd.Series(index=df_daily.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend.iloc[atr_period-1] = upper_band.iloc[atr_period-1]
    direction.iloc[atr_period-1] = 1
    
    # Calculate Supertrend iteratively
    for i in range(atr_period, len(df_daily)):
        if df_daily['close'].iloc[i] <= upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i-1]
        else:
            upper_band.iloc[i] = hl_avg.iloc[i] + (multiplier * atr.iloc[i])
            
        if df_daily['close'].iloc[i] >= lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i-1]
        else:
            lower_band.iloc[i] = hl_avg.iloc[i] - (multiplier * atr.iloc[i])
            
        if (supertrend.iloc[i-1] == upper_band.iloc[i-1] and 
            df_daily['close'].iloc[i] <= upper_band.iloc[i]):
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        elif (supertrend.iloc[i-1] == upper_band.iloc[i-1] and 
              df_daily['close'].iloc[i] > upper_band.iloc[i]):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif (supertrend.iloc[i-1] == lower_band.iloc[i-1] and 
              df_daily['close'].iloc[i] >= lower_band.iloc[i]):
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:  # supertrend.iloc[i-1] == lower_band.iloc[i-1] and close < lower_band
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
    
    # Convert to arrays
    supertrend_vals = supertrend.values
    direction_vals = direction.values
    
    # Handle NaN values at start
    for i in range(len(supertrend_vals)):
        if np.isnan(supertrend_vals[i]):
            if i > 0:
                supertrend_vals[i] = supertrend_vals[i-1]
                direction_vals[i] = direction_vals[i-1]
            else:
                supertrend_vals[i] = close[i]  # fallback
                direction_vals[i] = 1
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_daily, supertrend_vals)
    direction_aligned = align_htf_to_ltf(prices, df_daily, direction_vals)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: trend change or volume filter fails
            if direction_aligned[i] == -1 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: trend change or volume filter fails
            if direction_aligned[i] == 1 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price at or above Supertrend in uptrend with volume
            if (direction_aligned[i] == 1 and close[i] >= supertrend_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price at or below Supertrend in downtrend with volume
            elif (direction_aligned[i] == -1 and close[i] <= supertrend_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals