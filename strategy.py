#!/usr/bin/env python3
"""
6h_1d_supertrend_1w_trend_filter_v1
Hypothesis: In crypto markets, strong trends persist across multiple timeframes.
- Primary: 6h Supertrend (ATR=10, mult=3.0) for entry/exit signals
- Trend filter: 1d Supertrend direction to ensure alignment with daily trend
- Higher timeframe filter: 1w Supertrend to avoid counter-trend trades in strong weekly trends
- Entry only when all three timeframes agree on direction
- Position sizing: 0.25 for long, -0.25 for short
- Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_supertrend_1w_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator. Returns (supertrend, direction) where direction=1 for uptrend, -1 for downtrend."""
    # Calculate ATR
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    
    # Calculate basic upper and lower bands
    hl_avg = (pd.Series(high) + pd.Series(low)) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize final bands
    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    
    # Initialize Supertrend and direction
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    # Set first values
    supertrend.iloc[0] = 0.0
    direction.iloc[0] = 1  # Start with uptrend assumption
    
    for i in range(1, len(close)):
        # Update bands based on previous close
        if close.iloc[i-1] <= final_upper_band.iloc[i-1]:
            final_upper_band.iloc[i] = upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
            
        if close.iloc[i-1] >= final_lower_band.iloc[i-1]:
            final_lower_band.iloc[i] = lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Determine trend direction
        if close.iloc[i] > final_upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
        # Set Supertrend value
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = final_lower_band.iloc[i]
        else:
            supertrend.iloc[i] = final_upper_band.iloc[i]
    
    return supertrend.values, direction.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    st_1d, dir_1d = supertrend(high_1d, low_1d, close_1d, atr_period=10, multiplier=3.0)
    trend_1d_up = dir_1d == 1
    trend_1d_down = dir_1d == -1
    
    # Forward fill trend to handle any NaN
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().fillna(False).values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().fillna(False).values
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # Get 1w data for higher timeframe filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w Supertrend for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    st_1w, dir_1w = supertrend(high_1w, low_1w, close_1w, atr_period=10, multiplier=3.0)
    trend_1w_up = dir_1w == 1
    trend_1w_down = dir_1w == -1
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().fillna(False).values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().fillna(False).values
    
    # Align 1w trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # 6h Supertrend for entry signals
    st_6h, dir_6h = supertrend(high, low, close, atr_period=10, multiplier=3.0)
    st_uptrend = dir_6h == 1
    st_downtrend = dir_6h == -1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(st_6h[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 6h Supertrend turns down OR 1d trend turns down OR 1w trend turns down
            if st_downtrend[i] or trend_1d_down_aligned[i] or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: 6h Supertrend turns up OR 1d trend turns up OR 1w trend turns up
            if st_uptrend[i] or trend_1d_up_aligned[i] or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: 6h Supertrend uptrend + 1d uptrend + 1w uptrend
            if st_uptrend[i] and trend_1d_up_aligned[i] and trend_1w_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: 6h Supertrend downtrend + 1d downtrend + 1w downtrend
            elif st_downtrend[i] and trend_1d_down_aligned[i] and trend_1w_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals