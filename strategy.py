#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_TrendPullback_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w, 1)), np.absolute(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 4h ATR for stop loss (optional, not used in signal)
    tr_4h = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_1w_val = atr_1w_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend bias: long if price > EMA200, short if price < EMA200
        long_bias = price > ema200_1d_aligned[i]
        short_bias = price < ema200_1d_aligned[i]
        
        # Pullback condition: price within 0.5*ATR of EMA200 in direction of trend
        # For long: price slightly below EMA200 (pullback in uptrend)
        # For short: price slightly above EMA200 (pullback in downtrend)
        pullback_long = long_bias and (ema200_1d_aligned[i] - price) <= 0.5 * atr_1w_val
        pullback_short = short_bias and (price - ema200_1d_aligned[i]) <= 0.5 * atr_1w_val
        
        if position == 0:
            # Long: uptrend + pullback + volume
            if pullback_long and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + pullback + volume
            elif pullback_short and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks above EMA200 (trend acceleration) or volume dries up
            if price > ema200_1d_aligned[i] + atr_1w_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks below EMA200 (trend acceleration) or volume dries up
            if price < ema200_1d_aligned[i] - atr_1w_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals