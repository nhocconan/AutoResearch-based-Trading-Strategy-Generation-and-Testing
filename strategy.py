#!/usr/bin/env python3
"""
1h_4hTrend_1dPullback
Hypothesis: In strong 4h trends, buy pullbacks to 1h EMA20 during uptrends and sell rallies during downtrends.
Uses 4h EMA50 for trend direction and 1h EMA20 for entry timing. Volume spike filter avoids low-quality signals.
Session filter (08-20 UTC) reduces noise. Works in bull by buying dips; in bear by selling rallies.
Target: 15-35 trades/year (60-140 total over 4 years) to stay within fee limits.
"""
name = "1h_4hTrend_1dPullback"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 trend
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    for i in range(len(close_4h)):
        if i < 50:
            ema_4h[i] = np.nan
        elif i == 50:
            ema_4h[i] = np.mean(close_4h[0:50])
        else:
            ema_4h[i] = (close_4h[i] * 2 / (50 + 1)) + (ema_4h[i-1] * (49 / (50 + 1)))
    
    # EMA slope
    ema_slope_4h = np.full(len(close_4h), np.nan)
    for i in range(51, len(close_4h)):
        ema_slope_4h[i] = ema_4h[i] - ema_4h[i-1]
    
    # 1h EMA20 for entry
    ema_20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            ema_20[i] = np.nan
        elif i == 20:
            ema_20[i] = np.mean(close[0:20])
        else:
            ema_20[i] = (close[i] * 2 / (20 + 1)) + (ema_20[i-1] * (19 / (20 + 1)))
    
    # 1h ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # 1h volume MA(20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 4h indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slope_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Warmup
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(ema_4h_aligned[i]) or
            np.isnan(ema_slope_4h_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition
        uptrend = ema_slope_4h_aligned[i] > 0
        downtrend = ema_slope_4h_aligned[i] < 0
        
        # Entry conditions
        near_ema = np.abs(close[i] - ema_20[i]) < 0.3 * atr[i]
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if uptrend and close[i] > ema_20[i] and near_ema and vol_spike:
                # Long: buy pullback in uptrend
                signals[i] = 0.20
                position = 1
            elif downtrend and close[i] < ema_20[i] and near_ema and vol_spike:
                # Short: sell rally in downtrend
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: trend turns down or price breaks below EMA20
                if not uptrend or close[i] < ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: trend turns up or price breaks above EMA20
                if not downtrend or close[i] > ema_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals