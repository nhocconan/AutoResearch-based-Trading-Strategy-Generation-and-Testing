#!/usr/bin/env python3
# 1h_4h1d_ema_trend_pullback_v1
# Hypothesis: Uses 4h EMA50 for primary trend direction and 1d EMA200 for regime filter.
# Enters on 1h pullbacks to EMA20 with volume confirmation when aligned with higher timeframe trends.
# Long when: price > 4h EMA50, price > 1d EMA200, price pulls back to 1h EMA20, volume > 1.5x average.
# Short when: price < 4h EMA50, price < 1d EMA200, price pulls back to 1h EMA20, volume > 1.5x average.
# Uses 4 conditions max to avoid overtrading. Target: 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_ema_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA20 for pullback entries
    ema_period = 20
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d EMA200 for regime filter (bull/bear)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema_period, vol_ma_period, 50, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below 1h EMA20 or volume drops below average
            if close[i] < ema20[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price above 1h EMA20 or volume drops below average
            if close[i] > ema20[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: Price > 4h EMA50, Price > 1d EMA200, Pullback to 1h EMA20, Volume surge
            if (close[i] > ema50_4h_aligned[i] and 
                close[i] > ema200_1d_aligned[i] and 
                close[i] <= ema20[i] * 1.005 and  # Allow small tolerance for pullback
                vol_surge[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: Price < 4h EMA50, Price < 1d EMA200, Pullback to 1h EMA20, Volume surge
            elif (close[i] < ema50_4h_aligned[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  close[i] >= ema20[i] * 0.995 and  # Allow small tolerance for pullback
                  vol_surge[i]):
                position = -1
                signals[i] = -0.20
    
    return signals