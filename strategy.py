#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian20_Breakout_VolumeSpike
Hypothesis: Use 1d ADX to filter regimes (ADX>25 = trending, ADX<20 = range). In trending regime, trade 6h Donchian(20) breakouts with volume spike (>2.0x 20-bar MA). In range regime, fade Donchian extremes at weekly pivot levels (price > weekly pivot for short fade, price < weekly pivot for long fade) with volume spike. This adapts to both bull/bear markets by switching between trend following and mean reversion. Discrete sizing 0.25 limits fee drag. Target 12-25 trades/year.
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
    
    # Get 1d data for ADX and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) calculation
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
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
        plus_di[period] = np.nansum(plus_dm[1:period+1]) if period < len(plus_dm) else 0
        minus_di[period] = np.nansum(minus_dm[1:period+1]) if period < len(minus_dm) else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.zeros_like(high)
        denom = plus_di + minus_di
        denom[denom == 0] = 1e-10
        dx = 100 * np.abs(plus_di - minus_di) / denom
        
        adx = np.zeros_like(high)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period if 2*period < len(dx) else 0
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate ADX on 1d
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h (completed 1d bar only)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Weekly pivot from 1w data for range regime mean reversion
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), ADX (2*14=28)
    start_idx = max(20, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            if adx_val > 25:
                # Trending regime: Donchian breakout with volume spike
                long_setup = (close[i] > highest_high[i]) and volume_spike[i]
                short_setup = (close[i] < lowest_low[i]) and volume_spike[i]
            else:
                # Range regime: fade Donchian extremes at weekly pivot with volume spike
                long_setup = (close[i] < lowest_low[i]) and (close[i] < weekly_pivot_aligned[i]) and volume_spike[i]
                short_setup = (close[i] > highest_high[i]) and (close[i] > weekly_pivot_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if adx_val > 25:
                # Trending: exit on Donchian low break
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Range: exit on weekly pivot reversion or Donchian high
                if (close[i] > weekly_pivot_aligned[i]) or (close[i] > highest_high[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if adx_val > 25:
                # Trending: exit on Donchian high break
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Range: exit on weekly pivot reversion or Donchian low
                if (close[i] < weekly_pivot_aligned[i]) or (close[i] < lowest_low[i]):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_ADX_Regime_Donchian20_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0