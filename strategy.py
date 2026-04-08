#!/usr/bin/env python3
# 4h_pullback_v1
# Hypothesis: In trending markets, price pulls back to the 21-period EMA before continuing.
# Long when price > EMA21, price pulls back to touch EMA21 (within 0.5%), and 12h EMA21 slope > 0.
# Short when price < EMA21, price pulls back to touch EMA21 (within 0.5%), and 12h EMA21 slope < 0.
# Volume must be > 1.3x 20-period average for confirmation.
# Exit when price moves 1.5x ATR away from EMA21 in the opposite direction.
# Uses tight pullback entry to limit trades and reduce fee drag. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA21 for pullback
    ema_period = 21
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: 1.3x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    # Get 12h data for trend direction (EMA21 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema21_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(3, len(close_12h)):
        if not np.isnan(ema21_12h[i]) and not np.isnan(ema21_12h[i-3]):
            ema21_slope_12h[i] = ema21_12h[i] - ema21_12h[i-3]
    ema21_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(ema21_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price moves 1.5x ATR below EMA21
            if close[i] < ema21[i] - 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves 1.5x ATR above EMA21
            if close[i] > ema21[i] + 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Check for pullback to EMA21 (within 0.5%)
            pullback_zone = 0.005 * ema21[i]  # 0.5% of EMA value
            near_ema = abs(close[i] - ema21[i]) <= pullback_zone
            
            # Long entry: price > EMA21, near EMA (pullback), 12h EMA21 slope positive, volume surge
            if (close[i] > ema21[i] and 
                near_ema and 
                ema21_slope_12h_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < EMA21, near EMA (pullback), 12h EMA21 slope negative, volume surge
            elif (close[i] < ema21[i] and 
                  near_ema and 
                  ema21_slope_12h_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals