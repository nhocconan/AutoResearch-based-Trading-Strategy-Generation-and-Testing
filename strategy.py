#!/usr/bin/env python3
# 12h_1d_camarilla_volume_pivot_v1
# Hypothesis: 12h long/short entries at Camarilla pivot levels (H3/L3) from 1d timeframe,
# confirmed by volume spike (>1.5x 20-period average) and ADX trend filter (>25).
# Exits when price reaches opposite H4/L4 level or ADX drops below 20.
# Designed to capture intraday reversals at institutional levels while avoiding chop.
# Works in bull/bear by using mean-reversion at extremes with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1]) / period
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1]) / period
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1]) / period
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        adx = np.zeros_like(high)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros_like(high_1d)
    camarilla_l4 = np.zeros_like(low_1d)
    camarilla_h3 = np.zeros_like(high_1d)
    camarilla_l3 = np.zeros_like(low_1d)
    
    for i in range(1, len(high_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + range_ * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - range_ * 1.1 / 2
        camarilla_h3[i] = close_1d[i-1] + range_ * 1.1 / 4
        camarilla_l3[i] = close_1d[i-1] - range_ * 1.1 / 4
    
    # Align to 12h timeframe
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(adx[i]) or np.isnan(vol_avg[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 1.5 * vol_avg[i]
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        if position == 1:  # Long position
            if close[i] >= h4_12h[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if close[i] <= l4_12h[i] or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if vol_spike and strong_trend:
                if close[i] <= l3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= h3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals