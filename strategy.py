#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Triple_EMA_Filter_4hTrend_1hEntry"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1h EMA20 and EMA50 for entry timing
    close_series = pd.Series(close)
    ema20_1h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1h volume spike filter (volume > 2x 20-period MA)
    vol_series = pd.Series(volume)
    vol_ma20_1h = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d volume MA for context
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema20_1h[i]) or 
            np.isnan(ema50_1h[i]) or np.isnan(vol_ma20_1h[i]) or 
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1h volume > 2x 20-period MA
        vol_ok = volume[i] > 2.0 * vol_ma20_1h[i]
        
        # Trend filter: 4h EMA20 slope (using current and 3 periods ago)
        trend_up = ema20_4h_aligned[i] > ema20_4h_aligned[max(0, i-3)]
        trend_down = ema20_4h_aligned[i] < ema20_4h_aligned[max(0, i-3)]
        
        if position == 0:
            # Long: price > EMA20_1h and EMA20_1h > EMA50_1h, with volume and uptrend
            if close[i] > ema20_1h[i] and ema20_1h[i] > ema50_1h[i] and vol_ok and trend_up:
                signals[i] = 0.20
                position = 1
            # Short: price < EMA20_1h and EMA20_1h < EMA50_1h, with volume and downtrend
            elif close[i] < ema20_1h[i] and ema20_1h[i] < ema50_1h[i] and vol_ok and trend_down:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < EMA20_1h or trend turns down
            if close[i] < ema20_1h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > EMA20_1h or trend turns up
            if close[i] > ema20_1h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals