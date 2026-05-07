#!/usr/bin/env python3
name = "6h_PriceAction_1dResonance_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for resonance analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d price range and key levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d true range and ATR-like volatility measure
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First element has no previous close
    tr3[0] = tr1[0]
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # 1d exponential moving average for trend context
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h price position relative to 1d range
    # For each 6h bar, find the most recent completed 1d bar's range
    high_1d_recent = pd.Series(high_1d).expanding().apply(lambda x: x[-1] if len(x) > 0 else np.nan, raw=False).values
    low_1d_recent = pd.Series(low_1d).expanding().apply(lambda x: x[-1] if len(x) > 0 else np.nan, raw=False).values
    
    # Align the recent 1d high/low to 6b timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d_recent)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d_recent)
    
    # Calculate 6b volatility and momentum
    atr_6h = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_6h[0] = np.maximum(high[0] - low[0], np.maximum(np.abs(high[0] - close[0]), np.abs(low[0] - close[0])))
    
    # Price momentum over 3 periods
    price_change_3 = (close - np.roll(close, 3)) / np.roll(close, 3)
    price_change_3[:3] = 0
    
    # Volume surge detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(price_change_3[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 6b price position within 1d range
        range_size = high_1d_aligned[i] - low_1d_aligned[i]
        if range_size <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        price_position = (close[i] - low_1d_aligned[i]) / range_size  # 0 = at low, 1 = at high
        
        # Resonance conditions: price at extremes of 1d range with momentum and volume
        # Long setup: price near 1d low, bullish momentum, volume surge, above 1d EMA
        long_setup = (price_position < 0.2 and  # Near 1d low
                     price_change_3[i] > 0.01 and  # Positive short-term momentum
                     vol_ratio[i] > 1.8 and  # Volume surge
                     close[i] > ema_20_1d_aligned[i])  # Above 1d trend
        
        # Short setup: price near 1d high, bearish momentum, volume surge, below 1d EMA
        short_setup = (price_position > 0.8 and  # Near 1d high
                      price_change_3[i] < -0.01 and  # Negative short-term momentum
                      vol_ratio[i] > 1.8 and  # Volume surge
                      close[i] < ema_20_1d_aligned[i])  # Below 1d trend
        
        if position == 0:
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches middle of 1d range or momentum fails
            if price_position > 0.6 or price_change_3[i] < -0.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches middle of 1d range or momentum fails
            if price_position < 0.4 or price_change_3[i] > 0.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals