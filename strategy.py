#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4hTrend_1dVolume_OB"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Trend filter: 4h EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current 1d volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    volume_filter_1h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for volume MA and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_1h[i]) or np.isnan(volume_filter_1h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_4h_1h[i]
        vol_filter = volume_filter_1h[i]
        
        if position == 0:
            # Enter long: close > EMA50 (uptrend) + volume filter
            if close[i] > trend and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: close < EMA50 (downtrend) + volume filter
            elif close[i] < trend and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close < EMA50 (trend reversal)
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close > EMA50 (trend reversal)
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals