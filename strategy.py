#!/usr/bin/env python3
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
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA20
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h Bollinger Bands (20, 2) for mean reversion entries
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need 4h EMA and 1d volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_4h = ema_50_4h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        upper_bb = upper[i]
        lower_bb = lower[i]
        
        # Volume filter: volume > 1.3x 1d MA (avoid low-volume noise)
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_4h
        downtrend = close[i] < ema_4h
        
        # Entry conditions
        if position == 0:
            # Long: pullback to lower BB in uptrend + volume
            if uptrend and close[i] <= lower_bb and vol_filter:
                signals[i] = size
                position = 1
            # Short: pullback to upper BB in downtrend + volume
            elif downtrend and close[i] >= upper_bb and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: reversion to middle BB or trend change
            middle_bb = sma_20[i]
            if close[i] >= middle_bb or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: reversion to middle BB or trend change
            middle_bb = sma_20[i]
            if close[i] <= middle_bb or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_EMA50_4h_BB20_Pullback_Volume"
timeframe = "1h"
leverage = 1.0