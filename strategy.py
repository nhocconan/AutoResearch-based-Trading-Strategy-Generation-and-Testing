#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_12hTrend_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20, 2) for squeeze filter
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    
    # Bollinger Band Width percentile (50-period)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Volume confirmation: 20-period volume average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze filter: only trade when Bollinger Band Width is in lower 30th percentile
        squeeze_condition = bb_width_percentile[i] < 0.3
        
        # Volume confirmation: current volume > 1.5x average
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Break above upper Donchian with trend, volume, and squeeze
            if (close[i] > high_20[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_ok and squeeze_condition):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with trend, volume, and squeeze
            elif (close[i] < low_20[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_ok and squeeze_condition):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Break below lower Donchian or trend reversal
            if (close[i] < low_20[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Break above upper Donchian or trend reversal
            if (close[i] > high_20[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals