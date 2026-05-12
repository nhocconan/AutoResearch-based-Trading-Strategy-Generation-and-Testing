#!/usr/bin/env python3
"""
12h_Vortex_Trend_Reversal_Volume
Hypothesis: On 12h timeframe, Vortex indicator (VI+ and VI-) identifies trend direction and potential reversals. 
Long when VI+ crosses above VI- with VI+ > 1.1 and volume > 1.3x average; short when VI- crosses above VI+ with VI- > 1.1 and volume surge.
Uses 1w ADX < 25 to filter ranging markets (avoid false signals in low volatility). 
Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions to minimize fee drag.
Works in bull via trend continuation and bear via mean-reversion at trend exhaustion points.
"""

name = "12h_Vortex_Trend_Reversal_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for ADX filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate 1w ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(high)
        for i in range(len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx

    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)

    # Calculate Vortex Indicator on 12h data
    def calculate_vortex(high, low, close, period=14):
        vm_plus = np.zeros_like(high)
        vm_minus = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            vm_plus[i] = abs(high[i] - low[i-1])
            vm_minus[i] = abs(low[i] - high[i-1])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum over period
        vm_plus_sum = np.zeros_like(high)
        vm_minus_sum = np.zeros_like(high)
        tr_sum = np.zeros_like(high)
        
        for i in range(len(high)):
            if i < period:
                vm_plus_sum[i] = np.sum(vm_plus[1:i+1]) if i >= 1 else 0
                vm_minus_sum[i] = np.sum(vm_minus[1:i+1]) if i >= 1 else 0
                tr_sum[i] = np.sum(tr[1:i+1]) if i >= 1 else 0
            else:
                vm_plus_sum[i] = vm_plus_sum[i-1] - vm_plus[i-period+1] + vm_plus[i]
                vm_minus_sum[i] = vm_minus_sum[i-1] - vm_minus[i-period+1] + vm_minus[i]
                tr_sum[i] = tr_sum[i-1] - tr[i-period+1] + tr[i]
        
        vi_plus = np.where(tr_sum != 0, vm_plus_sum / tr_sum, 0)
        vi_minus = np.where(tr_sum != 0, vm_minus_sum / tr_sum, 0)
        return vi_plus, vi_minus

    vi_plus, vi_minus = calculate_vortex(high, low, close, 14)

    # Volume confirmation: 1.3x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_avg_20[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else 0
        else:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Get aligned values for current 12h bar
        adx = adx_1w_aligned[i]
        vip = vi_plus[i]
        vim = vi_minus[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is invalid
        if (np.isnan(adx) or np.isnan(vip) or np.isnan(vim) or 
            np.isnan(vol_avg_val) or vol_avg_val == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range filter: only trade when 1w ADX < 25 (avoid strong trends where Vortex whipsaws)
        if adx >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- AND VI+ > 1.1 AND volume surge
            if i > 14 and vi_plus[i-1] <= vi_minus[i-1] and vip > vim and vip > 1.1 and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ AND VI- > 1.1 AND volume surge
            elif i > 14 and vi_minus[i-1] <= vi_plus[i-1] and vim > vip and vim > 1.1 and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend reversal) OR VI+ drops below 1.0 (weakening trend)
            if vim > vip or vip < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend reversal) OR VI- drops below 1.0 (weakening trend)
            if vip > vim or vim < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals