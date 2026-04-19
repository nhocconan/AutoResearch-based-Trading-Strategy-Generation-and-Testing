#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian20_RangeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d ATR for volatility filter
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ADX for trend strength filter (avoid chop)
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    tr_1d_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_1d_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_1d_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14) + 1  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_1d_aligned[i] > 0.01 * price  # ATR > 1% of price
        
        # Trend strength filter: avoid choppy markets
        trend_filter = adx_aligned[i] > 20  # ADX > 20 indicates trending
        
        if position == 0:
            # Long breakout: price breaks above Donchian high + volatility + trend
            if price > donchian_high_aligned[i] and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Donchian low + volatility + trend
            elif price < donchian_low_aligned[i] and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Donchian midpoint or breakdown signal
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Donchian midpoint or breakout signal
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals