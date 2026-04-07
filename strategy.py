#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h/1d Trend + Volume Confirmation
# Hypothesis: Use 4h trend (EMA crossover) and 1d trend (price vs EMA200) for direction,
# with 1h volume confirmation for entry timing. This captures trend continuation
# while avoiding counter-trend trades. Works in bull (4h/1d uptrend + volume) and
# bear (4h/1d downtrend + volume). Target: 20-40 trades/year to minimize fee drag.
name = "1h_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA crossover: fast EMA12, slow EMA26
    close_4h = df_4h['close'].values
    ema12_4h = pd.Series(close_4h).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26_4h = pd.Series(close_4h).ewm(span=26, min_periods=26, adjust=False).mean().values
    trend_4h = ema12_4h - ema26_4h  # >0 = uptrend, <0 = downtrend
    trend_4h_1h = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    # Trend: price > EMA200 = uptrend, price < EMA200 = downtrend
    trend_1d = close_1d - ema200_1d
    trend_1d_1h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume confirmation: 1h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(trend_4h_1h[i]) or np.isnan(trend_1d_1h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Long: 4d uptrend + 1d uptrend + volume
        if trend_4h_1h[i] > 0 and trend_1d_1h[i] > 0 and vol_confirm:
            signals[i] = 0.20
        # Short: 4h downtrend + 1d downtrend + volume
        elif trend_4h_1h[i] < 0 and trend_1d_1h[i] < 0 and vol_confirm:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals