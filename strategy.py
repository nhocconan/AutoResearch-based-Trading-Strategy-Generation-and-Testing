#!/usr/bin/env python3
"""
1d_supertrend_1w_trend_filter_v2
Hypothesis: On daily timeframe, use Supertrend indicator with ATR multiplier 3 to capture strong trends, filtered by weekly Supertrend for higher timeframe alignment. Daily Supertrend provides entry/exit signals, while weekly Supertrend acts as a regime filter to avoid counter-trend trades. This approach reduces whipsaws and works in both bull and bear markets by following the dominant trend. Added volume confirmation to reduce false signals and lower trade frequency. Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_supertrend_1w_trend_filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend parameters
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(atr_period, n):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if i == atr_period:
            supertrend[i] = upper_band[i]
            direction[i] = -1  # start in downtrend
        else:
            if close[i] <= supertrend[i-1]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            
            # Adjust bands
            if direction[i] == 1:  # uptrend
                if lower_band[i] < lower_band[i-1]:
                    lower_band[i] = lower_band[i-1]
            else:  # downtrend
                if upper_band[i] > upper_band[i-1]:
                    upper_band[i] = upper_band[i-1]
            
            # Recalculate supertrend with adjusted bands
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly Supertrend for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < atr_period:
        return np.zeros(n)
    
    # Calculate weekly Supertrend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for weekly
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                            np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (atr_multiplier * atr_1w)
    lower_band_1w = hl2_1w - (atr_multiplier * atr_1w)
    
    supertrend_1w = np.full(len(df_1w), np.nan)
    direction_1w = np.full(len(df_1w), 1)
    
    for i in range(atr_period, len(df_1w)):
        if np.isnan(atr_1w[i]) or np.isnan(upper_band_1w[i]) or np.isnan(lower_band_1w[i]):
            continue
            
        if i == atr_period:
            supertrend_1w[i] = upper_band_1w[i]
            direction_1w[i] = -1
        else:
            if close_1w[i] <= supertrend_1w[i-1]:
                supertrend_1w[i] = upper_band_1w[i]
                direction_1w[i] = -1
            else:
                supertrend_1w[i] = lower_band_1w[i]
                direction_1w[i] = 1
            
            # Adjust bands
            if direction_1w[i] == 1:  # uptrend
                if lower_band_1w[i] < lower_band_1w[i-1]:
                    lower_band_1w[i] = lower_band_1w[i-1]
            else:  # downtrend
                if upper_band_1w[i] > upper_band_1w[i-1]:
                    upper_band_1w[i] = upper_band_1w[i-1]
            
            # Recalculate supertrend with adjusted bands
            if direction_1w[i] == 1:
                supertrend_1w[i] = lower_band_1w[i]
            else:
                supertrend_1w[i] = upper_band_1w[i]
    
    # Align weekly Supertrend to daily
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(atr_period, n):
        # Skip if data not available
        if (np.isnan(supertrend[i]) or np.isnan(close[i]) or 
            np.isnan(supertrend_1w_aligned[i]) or np.isnan(direction_1w_aligned[i]) or
            np.isnan(avg_volume[i]) or volume[i] < avg_volume[i] * 0.5):  # Volume filter: require at least 50% of average volume
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = direction_1w_aligned[i] == 1
        
        if position == 1:  # Long position
            # Exit: price closes below Supertrend (trend reversal)
            if close[i] <= supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Supertrend (trend reversal)
            if close[i] >= supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter if weekly trend aligns AND volume confirmation
            if weekly_uptrend and volume[i] > avg_volume[i]:
                # Long entry: price closes above Supertrend (uptrend start)
                if close[i] > supertrend[i]:
                    position = 1
                    signals[i] = 0.25
            elif not weekly_uptrend and volume[i] > avg_volume[i]:
                # Short entry: price closes below Supertrend (downtrend start)
                if close[i] < supertrend[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals