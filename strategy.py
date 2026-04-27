#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily ADX trend filter and volume spike confirmation
# Uses daily ADX(14) > 25 to identify trending markets and avoid whipsaws in ranging conditions.
# Donchian breakouts capture momentum in trending markets, with volume > 1.5x 20-period average
# confirming breakout strength. Works in both bull and bear markets by following the trend
# as indicated by ADX. Target: 20-30 trades/year to minimize fee decay while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # True Range
    tr = np.zeros(n_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, n_1d):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Directional Movement
    plus_dm = np.zeros(n_1d)
    minus_dm = np.zeros(n_1d)
    for i in range(1, n_1d):
        up = high_1d[i] - high_1d[i-1]
        down = low_1d[i-1] - low_1d[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(x[:period])
        # Subsequent values
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = np.full(n_1d, np.nan)
    minus_di = np.full(n_1d, np.nan)
    dx = np.full(n_1d, np.nan)
    for i in range(14, n_1d):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])
            if (plus_di[i] + minus_di[i]) > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX = smoothed DX
    adx = wilders_smoothing(dx, 14)
    
    # Align daily ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20-period) on 4h
    dc_period = 20
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    
    for i in range(dc_period, n):
        upper_dc[i] = np.max(high[i-dc_period:i])
        lower_dc[i] = np.min(low[i-dc_period:i])
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(dc_period, vol_period, 28)  # 28 for ADX calculation
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = price > upper_dc[i]
        breakout_down = price < lower_dc[i]
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: bullish breakout in trending market with volume
            if trending and breakout_up and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish breakout in trending market with volume
            elif trending and breakout_down and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian or ADX weakens
            if price < lower_dc[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian or ADX weakens
            if price > upper_dc[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0