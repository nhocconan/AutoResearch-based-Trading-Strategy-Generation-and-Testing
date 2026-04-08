#!/usr/bin/env python3
# 1h_trend_follow_4h1d_volume_v1
# Hypothesis: Trend following on 1h with 4h/1d trend filters and volume confirmation.
# Uses 4h EMA21 for trend direction, 1d EMA50 for higher timeframe filter,
# and volume > 1.3x average to confirm strength.
# Enters long when 1h close > 4h EMA21 > 1d EMA50 and volume surge.
# Enters short when 1h close < 4h EMA21 < 1d EMA50 and volume surge.
# Exits when trend breaks or volume drops.
# Target: 15-30 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_trend_follow_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h EMA21 for entry timing
    ema_period = 21
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Volume filter: 1.3x 24-period average (6 hours)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    # Get 4h data for trend direction (EMA21)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for higher timeframe filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema_period, vol_ma_period, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Trend breaks (price < 4h EMA21 or 4h EMA21 < 1d EMA50) or volume drops
            if (close[i] < ema21_4h_aligned[i] or 
                ema21_4h_aligned[i] < ema50_1d_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Trend breaks (price > 4h EMA21 or 4h EMA21 > 1d EMA50) or volume drops
            if (close[i] > ema21_4h_aligned[i] or 
                ema21_4h_aligned[i] > ema50_1d_aligned[i] or 
                volume[i] < vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: Price > 4h EMA21 > 1d EMA50 and volume surge
            if (close[i] > ema21_4h_aligned[i] and 
                ema21_4h_aligned[i] > ema50_1d_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: Price < 4h EMA21 < 1d EMA50 and volume surge
            elif (close[i] < ema21_4h_aligned[i] and 
                  ema21_4h_aligned[i] < ema50_1d_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.20
    
    return signals