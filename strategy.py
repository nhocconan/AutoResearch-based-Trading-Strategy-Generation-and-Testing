#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA200 trend filter and ATR(14) volatility filter.
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
- Long when price breaks above H3 with close > H3 AND 1w EMA200 uptrend (price > EMA200) AND ATR(14) > 0.3 * ATR(50)
- Short when price breaks below L3 with close < L3 AND 1w EMA200 downtrend (price < EMA200) AND ATR(14) > 0.3 * ATR(50)
- Exit when price reverses back inside the Camarilla H3/L3 range or volatility drops
- Designed to capture institutional breakouts with trend alignment and volatility filter
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.3 * atr_50)
    
    # Calculate prior 1d Camarilla H3/L3 levels (need 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla H3/L3: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_range = prior_high - prior_low
    h3_level = prior_close + 1.1 * camarilla_range / 2
    l3_level = prior_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Trend filter: price above/below 1w EMA200
    uptrend = close > ema_200_1w_aligned
    downtrend = close < ema_200_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 1)  # Need ATR, and prior 1d data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with close > H3 AND uptrend AND sufficient volatility
            if close[i] > h3_aligned[i] and uptrend[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with close < L3 AND downtrend AND sufficient volatility
            elif close[i] < l3_aligned[i] and downtrend[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back inside H3/L3 range OR volatility drops
            if close[i] < h3_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back inside H3/L3 range OR volatility drops
            if close[i] > l3_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1wEMA200_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0