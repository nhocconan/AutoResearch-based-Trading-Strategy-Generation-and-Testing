#!/usr/bin/env python3
"""
1h_4h1d_trend_volume_v1
Hypothesis: Use 1d EMA200 trend filter and 4h RSI momentum to guide 1h entries. Only trade in direction of higher timeframe trend. Volume surge confirms institutional participation. Target 15-37 trades/year by requiring confluence of 1d trend, 4h momentum, and volume spike on 1h. Designed to work in bull (follow trend) and bear (counter-trend bounces in range) markets via adaptive regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER (EMA200) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4H MOMENTUM (RSI14) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi14_4h = 100 - (100 / (1 + rs))
    rsi14_4h = rsi14_4h.fillna(50).values
    rsi14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi14_4h)
    
    # === 1H VOLUME SPIKE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi14_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require 1.5x average volume
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: 1d trend turns bearish OR 4h RSI overbought
            if close[i] <= ema200_1d_aligned[i] or rsi14_4h_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 1d trend turns bullish OR 4h RSI oversold
            if close[i] >= ema200_1d_aligned[i] or rsi14_4h_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish: price above 1d EMA200 AND 4h RSI rising from oversold
                if (close[i] > ema200_1d_aligned[i] and 
                    30 < rsi14_4h_aligned[i] < 50 and 
                    rsi14_4h_aligned[i] > rsi14_4h_aligned[i-1]):
                    position = 1
                    signals[i] = 0.20
                # Bearish: price below 1d EMA200 AND 4h RSI falling from overbought
                elif (close[i] < ema200_1d_aligned[i] and 
                      50 < rsi14_4h_aligned[i] < 70 and 
                      rsi14_4h_aligned[i] < rsi14_4h_aligned[i-1]):
                    position = -1
                    signals[i] = -0.20
    
    return signals