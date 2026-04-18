#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Reversion + Volume Spike + 12h ADX Trend Filter
Mean reversion at Camarilla pivot levels with volume confirmation and trend filter.
Long at S1 with bullish trend, short at R1 with bearish trend.
Designed for low trade frequency and high win rate in ranging markets.
"""

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
    
    # Get daily data for Camarilla pivots and 12h ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align pivot levels to 4h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h ADX for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.concatenate([[0], high_12h[1:] - high_12h[:-1]])
    down_move = np.concatenate([[0], low_12h[:-1] - low_12h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    # Initial values
    tr_sum[tr_period-1] = np.sum(tr[:tr_period])
    plus_dm_sum[tr_period-1] = np.sum(plus_dm[:tr_period])
    minus_dm_sum[tr_period-1] = np.sum(minus_dm[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / tr_period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / tr_period) + minus_dm[i]
    
    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    plus_di = 100 * plus_dm_sum / tr_sum_safe
    minus_di = 100 * minus_dm_sum / tr_sum_safe
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # ADX smoothing
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume spike detection (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 2*14)  # need enough history for ADX
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_value = adx_aligned[i]
        
        # Only trade when trend is strong enough (ADX > 25) or in ranging markets (ADX < 25)
        # But avoid choppy markets where ADX is very low (< 15)
        strong_trend = adx_value > 25
        ranging_market = 15 <= adx_value <= 25
        avoid_chop = adx_value >= 15
        
        if position == 0:
            # Long at S1 with volume spike and not in strong downtrend
            if (price <= s1_aligned[i] * 1.002 and  # Allow small slippage
                volume_spike[i] and
                not (strong_trend and price < pivot_aligned[i])):  # Not long in strong downtrend
                signals[i] = 0.25
                position = 1
            # Short at R1 with volume spike and not in strong uptrend
            elif (price >= r1_aligned[i] * 0.998 and  # Allow small slippage
                  volume_spike[i] and
                  not (strong_trend and price > pivot_aligned[i])):  # Not short in strong uptrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price reaches pivot (mean reversion target)
            # 2. Stop loss at S4 (unexpected breakdown)
            # 3. Strong bearish trend develops
            if (price >= pivot_aligned[i] * 0.999 or
                price <= s4_aligned[i] * 1.001 or
                (adx_value > 30 and price < pivot_aligned[i])):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price reaches pivot (mean reversion target)
            # 2. Stop loss at R4 (unexpected breakout)
            # 3. Strong bullish trend develops
            if (price <= pivot_aligned[i] * 1.001 or
                price >= r4_aligned[i] * 0.999 or
                (adx_value > 30 and price > pivot_aligned[i])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Reversion_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0