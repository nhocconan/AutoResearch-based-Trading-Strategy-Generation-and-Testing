#!/usr/bin/env python3
name = "6h_TrendShift_Trend"
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
    
    # Load 1D data ONCE for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1D EMA200 for long-term trend
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema200_1d = close_1d_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1D ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 6H EMA34 for trend shift
    close_s = pd.Series(close)
    ema34_6h = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 6H ATR(14) for volatility and momentum
    tr6h = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr6h[0] = high[0] - low[0]
    atr14_6h = pd.Series(tr6h).rolling(window=14, min_periods=14).mean().values
    
    # 6H Volume filter
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(atr14_6h[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend shift: price crossing EMA34 with momentum
        price_cross_up = close[i] > ema34_6h[i] and close[i-1] <= ema34_6h[i-1]
        price_cross_down = close[i] < ema34_6h[i] and close[i-1] >= ema34_6h[i-1]
        
        # Momentum: price change > 0.5 * ATR
        price_change = abs(close[i] - close[i-1])
        momentum_ok = price_change > 0.5 * atr14_6h[i]
        
        # Long-term trend filter
        long_term_up = close[i] > ema200_1d_aligned[i]
        long_term_down = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # LONG: bullish cross + momentum + uptrend
            if price_cross_up and momentum_ok and long_term_up and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish cross + momentum + downtrend
            elif price_cross_down and momentum_ok and long_term_down and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish cross or trend break
            if price_cross_down or not long_term_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish cross or trend break
            if price_cross_up or not long_term_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals