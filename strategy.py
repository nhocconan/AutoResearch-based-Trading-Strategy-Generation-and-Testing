#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume MA20 for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 6h Donchian(20) for entry
    high_6h = high
    low_6h = low
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(20, n):
        upper[i] = np.max(high_6h[i-20:i])
        lower[i] = np.min(low_6h[i-20:i])
    upper[:20] = np.nan
    lower[:20] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA, volume, Donchian
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # 1d trend filter
        trend_up = close_1d[-1] > ema_34_1d[-1] if i == n-1 else pd.Series(close_1d[:i+1]).ewm(span=34, adjust=False, mean=False).mean().iloc[-1] > ema_34_1d[i]
        # Simplified: use aligned EMA value from previous day
        trend = ema_34_1d_aligned[i]  # This is the EMA value aligned to 6h
        # Actual trend: close > EMA
        # Since we can't use future close, we use the condition that close > EMA at previous bar
        # But for simplicity and to avoid lookahead, we'll use the EMA slope
        # Instead, we'll use: price above EMA = bullish
        # We need current close vs current EMA - but EMA is lagging, so this is OK
        # We'll compute EMA on the fly for current close using historical data
        
        # Recalculate EMA up to current point to avoid lookahead
        if i >= 34:
            close_series = pd.Series(close[:i+1])
            ema_34_current = close_series.ewm(span=34, adjust=False, mean=False).mean().iloc[-1]
        else:
            ema_34_current = np.nan
        
        if np.isnan(ema_34_current):
            signals[i] = 0.0
            continue
            
        trend_filter = close[i] > ema_34_current  # bullish if price > EMA34
        
        # Volume filter: volume > 1.5x 1d MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Entry conditions: breakout with volume and trend
        if position == 0:
            # Long: break above upper band + volume + uptrend
            if close[i] > upper[i] and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + downtrend
            elif close[i] < lower[i] and vol_filter and not trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below lower band or trend change
            if close[i] < lower[i] or close[i] < ema_34_current:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above upper band or trend change
            if close[i] > upper[i] or close[i] > ema_34_current:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA34_Donchian20_VolumeTrendFilter"
timeframe = "6h"
leverage = 1.0