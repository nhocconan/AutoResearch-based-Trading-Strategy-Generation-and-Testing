# 12h_1d_Stochastic_Squeeze_V1
# Hypothesis: 12h timeframe with 1d stochastic squeeze and volatility breakout.
# Stochastic squeeze identifies low volatility periods, breakout captures momentum.
# Works in bull/bear by trading breakouts in direction of 1d trend (EMA50).
# Target: 20-40 trades/year with volume and trend filters to avoid overtrading.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Stochastic_Squeeze_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for stochastic and trend
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Stochastic Oscillator (14,3,3)
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * ((df_1d['close'] - low_14) / (high_14 - low_14 + 1e-10))
    k_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values  # %K
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values   # %D
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h
    k_percent_aligned = align_htf_to_ltf(prices, df_1d, k_percent)
    d_percent_aligned = align_htf_to_ltf(prices, df_1d, d_percent)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h volatility: ATR(10) for breakout threshold
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 12h Bollinger Band width for squeeze detection (20,2)
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = (upper - lower) / (sma + 1e-10)
    bb_width_ma = pd.Series(bb_width).rolling(window=10, min_periods=10).mean().values
    
    # Volume filter: current > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 10)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(k_percent_aligned[i]) or np.isnan(d_percent_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or \
           np.isnan(bb_width_ma[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        width = bb_width_ma[i]
        k = k_percent_aligned[i]
        d = d_percent_aligned[i]
        
        # Squeeze condition: Bollinger Band width below average (low volatility)
        squeeze = width < 0.8 * bb_width_ma[i]  # Using current width vs its MA
        
        # Breakout condition: price outside Bollinger Bands with volume
        breakout_up = price > upper[i] and vol > 1.5 * vol_ma
        breakout_down = price < lower[i] and vol > 1.5 * vol_ma
        
        # Trend bias from 1d EMA50
        bullish = price > ema50_1d_aligned[i]
        bearish = price < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: squeeze breakout up + bullish trend
            if squeeze and breakout_up and bullish:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout down + bearish trend
            elif squeeze and breakout_down and bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle band or breakdown
            if price < sma[i] or (price < lower[i] and vol > vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to middle band or breakup
            if price > sma[i] or (price > upper[i] and vol > vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals