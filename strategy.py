#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d Supertrend trend filter and volume confirmation.
# Long when price breaks above R4, price > 1d Supertrend, and volume > 1.8x 20-bar avg.
# Short when price breaks below S4, price < 1d Supertrend, and volume > 1.8x 20-bar avg.
# Exit when price reverts to the Camarilla pivot point (mean reversion).
# Uses 1d Supertrend for higher timeframe trend alignment, targeting 30-50 trades/year on 4h.
# Supertrend avoids whipsaws in ranging markets, volume confirmation reduces false breakouts.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "4h_Camarilla_R4_S4_Breakout_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    direction = np.full_like(close_1d, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after we have enough ATR data
    start_idx = atr_period
    for i in range(start_idx, len(close_1d)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if i == start_idx:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            # Calculate current Supertrend
            if supertrend[i-1] == upper_band[i-1]:
                supertrend[i] = lower_band[i] if close_1d[i] > upper_band[i-1] else upper_band[i]
                direction[i] = -1 if close_1d[i] > upper_band[i-1] else 1
            else:
                supertrend[i] = upper_band[i] if close_1d[i] < lower_band[i-1] else lower_band[i]
                direction[i] = 1 if close_1d[i] < lower_band[i-1] else -1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate Camarilla pivot levels from previous day (using same 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4 = close_1d + (range_1d * 1.1 / 2)  # R4 level
    s4 = close_1d - (range_1d * 1.1 / 2)  # S4 level
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, atr_period)  # warmup for Supertrend and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_supertrend = supertrend_aligned[i]
        curr_direction = direction_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R4, price > 1d Supertrend (uptrend), volume spike
            if (curr_close > curr_r4 and 
                curr_direction == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4, price < 1d Supertrend (downtrend), volume spike
            elif (curr_close < curr_s4 and 
                  curr_direction == -1 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to pivot point (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals