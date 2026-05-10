#!/usr/bin/env python3
# 4h_Pullback_Rebound_VolumeTrend
# Hypothesis: In strong trends (identified by 4h EMA50 alignment with 12h EMA200), price pulls back to the EMA20, finds support/resistance, and rebounds with volume confirmation. Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends. Uses tight entry conditions to limit trades and avoid fee drag.

name = "4h_Pullback_Rebound_VolumeTrend"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    ema_200_12h = pd.Series(df_12h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 4h EMA50 and EMA20 for pullback identification
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA200 (200), 4h EMA50 (50), EMA20 (20), volume MA (20)
    start_idx = max(200, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price must be on same side of 12h EMA200 as 4h EMA50
        price_vs_12h_ema = close[i] > ema_200_12h_aligned[i]
        ema50_vs_12h_ema = ema_50[i] > ema_200_12h_aligned[i]
        trend_aligned = price_vs_12h_ema == ema50_vs_12h_ema
        
        # Pullback condition: price near EMA20 (within 1%)
        near_ema20 = abs(close[i] - ema_20[i]) / ema_20[i] < 0.01
        
        # Volume confirmation (>1.5x average)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend alignment + pullback to EMA20 + volume
            if trend_aligned and close[i] > ema_200_12h_aligned[i] and near_ema20 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend alignment + pullback to EMA20 + volume
            elif trend_aligned and close[i] < ema_200_12h_aligned[i] and near_ema20 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves significantly above EMA20
            if not trend_aligned or close[i] < ema_200_12h_aligned[i] or close[i] > ema_20[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price moves significantly below EMA20
            if not trend_aligned or close[i] > ema_200_12h_aligned[i] or close[i] < ema_20[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals