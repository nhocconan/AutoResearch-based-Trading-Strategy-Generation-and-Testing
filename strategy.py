#!/usr/bin/env python3
name = "6h_1d_OrderBlock_TrendFollow_12hVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Order Blocks (bullish: strong up candle after consolidation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Bullish OB: previous candle bearish, current bullish with strong body
    body_1d = np.abs(close_1d - open_1d)
    range_1d = high_1d - low_1d
    # Avoid division by zero
    body_ratio = np.where(range_1d > 0, body_1d / range_1d, 0)
    prev_bearish = close_1d < open_1d
    curr_bullish = close_1d > open_1d
    strong_body = body_ratio > 0.6
    bullish_ob = prev_bearish & curr_bullish & strong_body
    
    # Bearish OB: previous candle bullish, current bearish with strong body
    prev_bullish = close_1d > open_1d
    curr_bearish = close_1d < open_1d
    bearish_ob = prev_bullish & curr_bearish & strong_body
    
    # OB levels: use the candle's range
    ob_high = np.where(bullish_ob | bearish_ob, high_1d, np.nan)
    ob_low = np.where(bullish_ob | bearish_ob, low_1d, np.nan)
    
    # Forward fill OB levels until next OB
    ob_high_series = pd.Series(ob_high)
    ob_low_series = pd.Series(ob_low)
    ob_high_ffill = ob_high_series.ffill().values
    ob_low_ffill = ob_low_series.ffill().values
    
    # 12h Volume filter: above average volume
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values  # 24 * 12h = 12 days
    vol_ratio = volume_12h / vol_ma
    
    # Align to 6h
    ob_high_6h = align_htf_to_ltf(prices, df_1d, ob_high_ffill)
    ob_low_6h = align_htf_to_ltf(prices, df_1d, ob_low_ffill)
    vol_ratio_6h = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    # 6h trend: EMA(20) vs EMA(50)
    ema_fast = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ob_high_6h[i]) or np.isnan(ob_low_6h[i]) or np.isnan(vol_ratio_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: only trade when volume is above average
        vol_filter = vol_ratio_6h[i] > 1.2
        
        if position == 0:
            # Long: price above bullish OB + uptrend + volume filter
            if (close[i] > ob_high_6h[i] and 
                trend_up[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below bearish OB + downtrend + volume filter
            elif (close[i] < ob_low_6h[i] and 
                  trend_down[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below bearish OB or trend change
            if close[i] < ob_low_6h[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above bullish OB or trend change
            if close[i] > ob_high_6h[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals