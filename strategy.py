#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for volatility filter (14-period)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 50-period SMA of ATR for volatility normalization
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels for each 1d bar
    hl_range = high_1d - low_1d
    r4_level = close_1d + 1.1 * hl_range * 1.5
    r3_level = close_1d + 1.1 * hl_range * 1.25
    s3_level = close_1d - 1.1 * hl_range * 1.25
    s4_level = close_1d - 1.1 * hl_range * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(atr_ma_50[i]) or np.isnan(r3_level[i]) or 
            np.isnan(r4_level[i]) or np.isnan(s3_level[i]) or 
            np.isnan(s4_level[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 6h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)[i]
        r3_level_aligned = align_htf_to_ltf(prices, df_1d, r3_level)[i]
        r4_level_aligned = align_htf_to_ltf(prices, df_1d, r4_level)[i]
        s3_level_aligned = align_htf_to_ltf(prices, df_1d, s3_level)[i]
        s4_level_aligned = align_htf_to_ltf(prices, df_1d, s4_level)[i]
        
        # Volatility filter: avoid extremely low volatility (choppy) conditions
        volatility_filter = atr_1d_aligned > (atr_ma_50_aligned * 0.8)
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 level OR trend reversal
            if close[i] < s3_level_aligned or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 level OR trend reversal
            if close[i] > r3_level_aligned or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above R4 level + uptrend + volatility filter (breakout continuation)
            if close[i] > r4_level_aligned and uptrend and volatility_filter:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 level + downtrend + volatility filter (breakout continuation)
            elif close[i] < s4_level_aligned and downtrend and volatility_filter:
                position = -1
                signals[i] = -0.25
            # Long: price touches S3 level + downtrend + volatility filter (fade at support)
            elif close[i] <= s3_level_aligned and not uptrend and volatility_filter:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 level + uptrend + volatility filter (fade at resistance)
            elif close[i] >= r3_level_aligned and not downtrend and volatility_filter:
                position = -1
                signals[i] = -0.25
    
    return signals