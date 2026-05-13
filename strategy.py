#!/usr/bin/env python3
name = "4h_DonchianBreakout_VolumeTrend_v3"
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
    
    # Load 12H data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12H for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels on 4H
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12H EMA50
        price_above_trend = close[i] > ema50_12h_aligned[i]
        price_below_trend = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # LONG: Donchian breakout + volume + uptrend
            if close[i] > upper[i] and vol_ok and price_above_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown + volume + downtrend
            elif close[i] < lower[i] and vol_ok and price_below_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Donchian breakdown or trend reversal
            if close[i] < lower[i] or not price_above_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Donchian breakout or trend reversal
            if close[i] > upper[i] or not price_below_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals