#!/usr/bin/env python3
"""
6h_12h_1d_VWAP_Deviation_MeanReversion
Hypothesis: In 6h timeframe, price often deviates from 12h VWAP during short-term momentum bursts but reverts to the mean. 
We take mean-reversion trades when price deviates >1.5 standard deviations from 12h VWAP, confirmed by 1d trend filter (price above/below 20 EMA).
Works in both bull and bear markets because mean reversion occurs in all regimes, and trend filter ensures we trade with higher timeframe momentum.
Target: 15-30 trades/year on 6h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP on 12h: typical price * volume / cumulative volume
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_values = vwap_12h.values
    
    # Calculate standard deviation of price deviation from VWAP over last 20 periods
    price_deviation_12h = (typical_price_12h - vwap_12h).values
    # Rolling std of deviation
    deviation_std_20 = pd.Series(price_deviation_12h).rolling(window=20, min_periods=20).std()
    deviation_std_20_values = deviation_std_20.values
    
    # Get 1d data for trend filter: 20 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all data to 6h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h_values)
    deviation_std_20_aligned = align_htf_to_ltf(prices, df_12h, deviation_std_20_values)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(deviation_std_20_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Typical price for current 6h bar
        typical_price = (high[i] + low[i] + close[i]) / 3.0
        
        # Deviation from 12h VWAP
        deviation = typical_price - vwap_12h_aligned[i]
        
        # Avoid division by zero
        if deviation_std_20_aligned[i] == 0:
            signals[i] = 0.0
            continue
            
        # Z-score of deviation
        z_score = deviation / deviation_std_20_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA20, bearish if price < EMA20
        # Note: we use close price for trend determination
        trend_bullish = close[i] > ema_20_1d_aligned[i]
        
        # Mean reversion conditions:
        # Long: price significantly below VWAP (z-score < -1.5) in uptrend
        # Short: price significantly above VWAP (z-score > 1.5) in downtrend
        long_condition = (z_score < -1.5) and trend_bullish
        short_condition = (z_score > 1.5) and (not trend_bullish)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_VWAP_Deviation_MeanReversion"
timeframe = "6h"
leverage = 1.0