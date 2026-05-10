#!/usr/bin/env python3
# 4H_Donchian_Breakout_VolumeTrend
# Hypothesis: 4-hour Donchian channel breakouts with volume confirmation and higher-timeframe trend filter (12h EMA50) capture trends while minimizing false breakouts. Works in bull/bear by following 12h trend direction. Target: 20-40 trades/year per symbol.

name = "4H_Donchian_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend_12h = close_12h > ema50_12h
    bearish_trend_12h = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    bullish_aligned = align_htf_to_ltf(prices, df_12h, bullish_trend_12h.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_12h, bearish_trend_12h.astype(float))
    
    # Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume ratio (current vs 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price breaks above upper Donchian + volume confirmation
            if bullish and close[i] > upper[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below lower Donchian + volume confirmation
            elif bearish and close[i] < lower[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price breaks below lower Donchian
            if bearish or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price breaks above upper Donchian
            if bullish or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals