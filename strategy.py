#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WMA_Breakout_VolumeTrend_1wFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly 34-period WMA for trend
    weights = np.arange(1, 35)
    wma_1w = np.convolve(close_1w, weights, mode='full')[:len(close_1w)] / weights.sum()
    wma_1w = np.pad(wma_1w, (34-1, 0), mode='edge')[:len(close_1w)]
    wma_1w_aligned = align_htf_to_ltf(prices, df_1w, wma_1w)
    
    # Daily 20-period Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for volatility and stop
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x 20-day average
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * avg_vol
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(wma_1w_aligned[i]) or np.isnan(donch_high[i]) or \
           np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + weekly uptrend
            if high[i] > donch_high[i-1] and volume_filter[i] and price > wma_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume + weekly downtrend
            elif low[i] < donch_low[i-1] and volume_filter[i] and price < wma_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian low or 2x ATR stop
            if close[i] < donch_low[i] or close[i] < close[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian high or 2x ATR stop
            if close[i] > donch_high[i] or close[i] > close[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals