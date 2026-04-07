#!/usr/bin/env python3
"""
12h_volatility_squeeze_breakout
Hypothesis: Combines Bollinger Band squeeze (low volatility) with Donchian breakout and volume confirmation.
In low volatility regimes, price builds energy for explosive moves. We enter on breakouts from the squeeze
with volume confirmation, using higher timeframe (1d/1w) trend filter to avoid counter-trend trades.
Works in both bull and bear markets by capturing volatility expansion phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_squeeze_breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for squeeze detection
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (20-period lookback) for squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Donchian channels (20-period) for breakout
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Higher timeframe filters: 1d and 1w trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # ADX (14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_sum = np.sum(plus_dm[1:period])
            minus_dm_sum = np.sum(minus_dm[1:period])
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * (plus_dm_sum / atr[i]) if atr[i] != 0 else 0
                minus_di[i] = 100 * (minus_dm_sum / atr[i]) if atr[i] != 0 else 0
                plus_dm_sum = plus_dm_sum - plus_dm[i-period+1] + plus_dm[i]
                minus_dm_sum = minus_dm_sum - minus_dm[i-period+1] + minus_dm[i]
        
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        bb_squeeze = bb_width_percentile[i] < 20  # Bollinger Band width in lowest 20%
        vol_expansion = bb_width_percentile[i] > 80  # Bollinger Band width in highest 20% (exit condition)
        vol_confirmed = volume[i] > vol_ma[i] * 1.5  # Volume 1.5x average
        uptrend_filter = (close[i] > ema_1d_aligned[i]) and (close[i] > ema_1w_aligned[i])
        downtrend_filter = (close[i] < ema_1d_aligned[i]) and (close[i] < ema_1w_aligned[i])
        strong_trend = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit: volatility expansion or breakdown below Donchian low
            if vol_expansion or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: volatility expansion or breakout above Donchian high
            if vol_expansion or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian breakout with volume, in uptrend or strong trend
            if close[i] > donchian_high[i] and vol_confirmed and (uptrend_filter or strong_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakdown with volume, in downtrend or strong trend
            elif close[i] < donchian_low[i] and vol_confirmed and (downtrend_filter or strong_trend):
                position = -1
                signals[i] = -0.25
    
    return signals