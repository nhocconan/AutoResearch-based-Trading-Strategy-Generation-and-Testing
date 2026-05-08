#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and ADX > 25.
# Short when price breaks below Donchian(20) low with volume > 1.5x average and ADX > 25.
# Exit when price crosses the opposite Donchian band or ADX < 20 (trend weakening).
# Works in trending markets by capturing breakouts; avoids false signals in low volatility.
# Target: 20-50 trades per year to minimize fee drag.

name = "4h_DonchianBreakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) on 4h
    def donchian(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper_dc, lower_dc = donchian(high, low, 20)
    
    # Volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ADX(14) on 1d
    def adx(high, low, close, period=14):
        tr = np.zeros_like(high)
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            dm_plus[i] = max(high[i] - high[i-1], 0)
            dm_minus[i] = max(low[i-1] - low[i], 0)
        
        # Smooth TR, DM+, DM-
        tr_smooth = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        tr_smooth[period-1] = np.sum(tr[:period])
        dm_plus_smooth[period-1] = np.sum(dm_plus[:period])
        dm_minus_smooth[period-1] = np.sum(dm_minus[:period])
        
        for i in range(period, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # DI+ and DI-
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) != 0, dx, 0)
        
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 19, 2*14-1)  # Donchian(20), vol MA(20), ADX(14) warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume confirmation and strong trend
            if close[i] > upper_dc[i] and volume[i] > 1.5 * vol_ma[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.30
                position = 1
            elif close[i] < lower_dc[i] and volume[i] > 1.5 * vol_ma[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price crosses lower Donchian or trend weakens
            if close[i] < lower_dc[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses upper Donchian or trend weakens
            if close[i] > upper_dc[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals