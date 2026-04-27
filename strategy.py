#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + 1d ADX Trend Filter
Hypothesis: Donchian(20) breakouts on 12h capture strong momentum moves.
Volume spike (2x avg volume) confirms institutional participation.
1d ADX > 25 filters for trending markets only, avoiding chop.
Works in bull (breakouts up) and bear (breakouts down) by trading both directions.
Targets 15-35 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 14-period ADX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        if len(tr) >= period:
            # Initial average
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_avg = np.nanmean(plus_dm[1:period+1])
            minus_dm_avg = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_avg = (plus_dm_avg * (period-1) + plus_dm[i]) / period
                minus_dm_avg = (minus_dm_avg * (period-1) + minus_dm[i]) / period
                
                if atr[i] != 0:
                    plus_di[i] = 100 * plus_dm_avg / atr[i]
                    minus_di[i] = 100 * minus_dm_avg / atr[i]
            
            # DX and ADX
            dx = np.full_like(tr, np.nan)
            for i in range(period+1, len(tr)):
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            adx = np.full_like(tr, np.nan)
            if len(tr) >= 2*period:
                adx[2*period] = np.nanmean(dx[period+1:2*period+1])
                for i in range(2*period+1, len(tr)):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ADX, volume MA
    start_idx = max(donchian_period - 1, 28, 19)  # ADX needs ~28 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        
        # ADX filter: only trade when trending (ADX > 25)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and trend
            if (price > upper_channel[i] and volume_spike[i] and trend_filter):
                signals[i] = size
                position = 1
            # Short: break below lower Donchian with volume spike and trend
            elif (price < lower_channel[i] and volume_spike[i] and trend_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below midpoint)
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above midpoint)
            midpoint = (upper_channel[i] + lower_channel[i]) / 2
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_ADX25"
timeframe = "12h"
leverage = 1.0