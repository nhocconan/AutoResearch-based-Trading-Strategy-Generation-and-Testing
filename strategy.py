#!/usr/bin/env python3
"""
1d_1w_fibonacci_extension_volume_v1
Hypothesis: On daily timeframe, use weekly Fibonacci extension levels (127.2%, 161.8%) from weekly swing high/low for breakout entries in trending markets, and Fibonacci retracement levels (38.2%, 61.8%) for mean reversion in ranging markets. Volume confirmation filters false signals. Weekly trend filter (EMA200) adapts to bull/bear regimes. Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag while capturing major moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_fibonacci_extension_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Fibonacci levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly swing points (using 20-period lookback for pivot detection)
    # We'll use rolling max/min to identify significant swing points
    window = 20
    high_max = pd.Series(high_1w).rolling(window=window, min_periods=window).max().values
    low_min = pd.Series(low_1w).rolling(window=window, min_periods=window).min().values
    
    # Calculate Fibonacci levels based on recent swing
    # For extension: use swing from low_min to high_max
    # For retracement: use swing from high_max to low_min
    diff = high_max - low_min
    
    # Fibonacci extension levels (for breakouts)
    ext_127 = low_min + diff * 1.272
    ext_162 = low_min + diff * 1.618
    
    # Fibonacci retracement levels (for mean reversion)
    ret_382 = high_max - diff * 0.382
    ret_618 = high_max - diff * 0.618
    
    # Weekly EMA200 for trend filter (slow MA for regime detection)
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    
    # Align weekly levels to daily timeframe
    ext_127_d = align_htf_to_ltf(prices, df_1w, ext_127)
    ext_162_d = align_htf_to_ltf(prices, df_1w, ext_162)
    ret_382_d = align_htf_to_ltf(prices, df_1w, ret_382)
    ret_618_d = align_htf_to_ltf(prices, df_1w, ret_618)
    ema200_1w_d = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if required data not available
        if (np.isnan(ext_127_d[i]) or np.isnan(ext_162_d[i]) or 
            np.isnan(ret_382_d[i]) or np.isnan(ret_618_d[i]) or 
            np.isnan(ema200_1w_d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 61.8% retracement (mean reversion fail) OR
            # price reaches 161.8% extension and weekly trend is weak (close < EMA200)
            if close[i] < ret_618_d[i] or (close[i] > ext_162_d[i] and close[i] < ema200_1w_d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above 38.2% retracement (mean reversion fail) OR
            # price reaches 161.8% extension (from high) and weekly trend is strong (close > EMA200)
            # For shorts, we invert the extension logic
            if close[i] > ret_382_d[i] or (close[i] < ext_162_d[i] and close[i] > ema200_1w_d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at 61.8% retracement in uptrend (price > EMA200)
            if (close[i] <= ret_618_d[i] and 
                vol_confirm and 
                close[i] > ema200_1w_d[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at 38.2% retracement in downtrend (price < EMA200)
            elif (close[i] >= ret_382_d[i] and 
                  vol_confirm and 
                  close[i] < ema200_1w_d[i]):
                position = -1
                signals[i] = -0.25
            # Breakout long above 127.2% extension in uptrend
            elif (close[i] >= ext_127_d[i] and 
                  vol_confirm and 
                  close[i] > ema200_1w_d[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below 127.2% extension (calculated from high) in downtrend
            # For short breakout, we use extension from high downward
            elif (close[i] <= (high_max[i] - diff[i] * 0.272) and  # Equivalent to -27.2% extension from high
                  vol_confirm and 
                  close[i] < ema200_1w_d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals