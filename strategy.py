#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Trend_Follow_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA34 for trend
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 1d EMA200 for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1h volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 200, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filters
        bullish_4h = ema34_4h_aligned[i] > price  # Price above EMA34 = bullish
        bearish_4h = ema34_4h_aligned[i] < price  # Price below EMA34 = bearish
        bullish_1d = ema200_1d_aligned[i] < price  # Price above EMA200 = bullish
        bearish_1d = ema200_1d_aligned[i] > price  # Price below EMA200 = bearish
        
        if position == 0:
            # Long: price above both 4h and 1d EMAs with volume
            if bullish_4h and bullish_1d and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: price below both 4h and 1d EMAs with volume
            elif bearish_4h and bearish_1d and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price crosses below 4h EMA or loses 1d trend
            if not bullish_4h or not bullish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price crosses above 4h EMA or loses 1d trend
            if not bearish_4h or not bearish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals