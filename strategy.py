#!/usr/bin/env python3
"""
4h_rsi_pullback_volume_v1
Hypothesis: RSI pullbacks with volume confirmation on 4h timeframe. Works in bull and bear markets by trading pullbacks to the mean during trends.
- Long when RSI < 30, price above 20-period EMA, and volume > 1.5x average
- Short when RSI > 70, price below 20-period EMA, and volume > 1.5x average
- Uses 12h EMA trend filter to avoid counter-trend trades
- Targets ~25 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_volume_v1"
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
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        # Initial average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 20-period EMA for dynamic support/resistance
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h_series = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_12h = ema_12h_series.values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 19) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 or price below EMA20
            if rsi[i] > 50 or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 or price above EMA20
            if rsi[i] < 50 or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30, price above EMA20, volume surge, 12h trend bullish
            if (rsi[i] < 30 and 
                close[i] > ema_20[i] and 
                vol_surge[i] and 
                close[i] > ema_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70, price below EMA20, volume surge, 12h trend bearish
            elif (rsi[i] > 70 and 
                  close[i] < ema_20[i] and 
                  vol_surge[i] and 
                  close[i] < ema_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals