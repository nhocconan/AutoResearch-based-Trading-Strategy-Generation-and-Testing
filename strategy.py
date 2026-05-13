#!/usr/bin/env python3
name = "6h_Liquidity_Refill_and_Trend"
timeframe = "6h"
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
    
    # Load 12h data ONCE for trend and liquidity zones
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h ATR(14) for volatility-based liquidity zones
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Identify liquidity zones: recent 12h high/low ± 0.5*ATR
    # We'll use rolling max/min of high/low over last 3 12h bars (~1.5 days)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.nanmax(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.nanmin(arr[i-window+1:i+1])
        return res
    
    # Recent 3-bar high/low for liquidity zones
    recent_high_12h = rolling_max(high_12h, 3)
    recent_low_12h = rolling_min(low_12h, 3)
    
    # Liquidity zones: recent swing points ± liquidity buffer
    liquidity_buffer = 0.5 * atr_12h
    liquidity_high_zone = recent_high_12h + liquidity_buffer
    liquidity_low_zone = recent_low_12h - liquidity_buffer
    
    # Align 12h indicators to 6h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    liquidity_high_zone_aligned = align_htf_to_ltf(prices, df_12h, liquidity_high_zone)
    liquidity_low_zone_aligned = align_htf_to_ltf(prices, df_12h, liquidity_low_zone)
    
    # 6h volume spike detection (20-period average)
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(liquidity_high_zone_aligned[i]) or 
            np.isnan(liquidity_low_zone_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation: significant spike
        vol_spike = vol_ratio[i] > 2.0
        
        # Price near liquidity zones (within 0.3*ATR buffer)
        near_liquidity_high = close[i] >= liquidity_high_zone_aligned[i] - (0.3 * atr_12h[i] if not np.isnan(atr_12h[i]) else np.inf)
        near_liquidity_low = close[i] <= liquidity_low_zone_aligned[i] + (0.3 * atr_12h[i] if not np.isnan(atr_12h[i]) else np.inf)
        
        if position == 0:
            # LONG: Uptrend + price near liquidity high (stop hunt) + volume spike
            if uptrend and near_liquidity_high and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price near liquidity low (stop hunt) + volume spike
            elif downtrend and near_liquidity_low and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or overextension above liquidity
            if not uptrend or close[i] > liquidity_high_zone_aligned[i] + (atr_12h[i] if not np.isnan(atr_12h[i]) else np.inf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or overextension below liquidity
            if not downtrend or close[i] < liquidity_low_zone_aligned[i] - (atr_12h[i] if not np.isnan(atr_12h[i]) else np.inf):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals