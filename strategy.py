#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Confirmation + 1d ADX Trend Filter
# Uses Donchian(20) breakouts for entry, volume > 1.5x 20-bar median for confirmation,
# and 1d ADX > 20 to filter for trending markets only. Avoids ranging markets where
# breakouts fail. Discrete sizing (0.25) limits trade frequency to ~25-40/year.
# Works in bull (breakouts continue) and bear (breakdowns) via symmetric long/short logic.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.absolute(high_1d[1:] - close_1d[:-1]), 
                    np.absolute(low_1d[1:] - close_1d[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=1).min().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Donchian breakout above upper band, volume spike, ADX > 20 (trending)
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold[i] and 
            adx_1d_aligned[i] > 20):
            signals[i] = 0.25
        
        # Short: Donchian breakdown below lower band, volume spike, ADX > 20 (trending)
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold[i] and 
              adx_1d_aligned[i] > 20):
            signals[i] = -0.25
        
        # Exit: Donchian reverse signal or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < lowest_low[i] or adx_1d_aligned[i] <= 20)) or
               (signals[i-1] == -0.25 and (close[i] > highest_high[i] or adx_1d_aligned[i] <= 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0