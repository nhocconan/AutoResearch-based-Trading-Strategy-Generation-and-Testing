#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Fractal breakout + volume confirmation + chop regime filter
# Williams Fractals identify key swing highs/lows from 1d timeframe
# Long when price breaks above recent bullish fractal with volume expansion and chop < 61.8 (trending)
# Short when price breaks below recent bearish fractal with volume expansion and chop < 61.8
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, chop filter avoids whipsaws in ranging markets

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals on 1d
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-1] > high_1d[i-3] and
            high_1d[i-1] > high_1d[i+1]):
            bearish_fractal[i-1] = high_1d[i-1]  # Value at the fractal point
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-1] < low_1d[i-3] and
            low_1d[i-1] < low_1d[i+1]):
            bullish_fractal[i-1] = low_1d[i-1]  # Value at the fractal point
    
    # Calculate 1d chopiness index (14-period) for regime filter
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr_1d = np.full(n_1d, np.nan)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, n_1d):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    def rolling_sum(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).sum().values
    
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    sum_tr_14 = rolling_sum(tr_1d, 14)
    max_high_14 = rolling_max(high_1d, 14)
    min_low_14 = rolling_min(low_1d, 14)
    
    chop_1d = 100 * np.log10(sum_tr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Chop regime filter: only trade when trending (chop < 61.8)
        chop_filter = chop_1d_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below recent bullish fractal
            if close[i] < bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above recent bearish fractal
            if close[i] > bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on fractal breakout with volume confirmation and chop filter
            if (close[i] > bullish_fractal_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                position = 1
                signals[i] = 0.25
            elif (close[i] < bearish_fractal_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                position = -1
                signals[i] = -0.25
    
    return signals