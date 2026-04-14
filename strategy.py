#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout + volume confirmation + ADX filter.
# Long when price breaks above 1d Donchian upper band AND 4h volume > 1.5x 20-period average AND ADX > 25.
# Short when price breaks below 1d Donchian lower band AND 4h volume > 1.5x 20-period average AND ADX > 25.
# Exit when price crosses back inside the 1d Donchian channel.
# Volume confirmation filters false breakouts, ADX ensures trending markets.
# Target: 20-40 trades/year to minimize fee drag and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channel
    donchian_up = np.full_like(high_1d, np.nan)
    donchian_down = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        donchian_up[i] = np.max(high_1d[i-19:i+1])
        donchian_down[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate ADX on 4h data
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(high)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    adx = np.zeros_like(high)
    
    period = 14
    alpha = 1.0 / period
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_sum = np.sum(plus_dm[1:period+1])
    minus_dm_sum = np.sum(minus_dm[1:period+1])
    plus_di[period] = 100 * plus_dm_sum / atr[period] if atr[period] > 0 else 0
    minus_di[period] = 100 * minus_dm_sum / atr[period] if atr[period] > 0 else 0
    dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period]) if (plus_di[period] + minus_di[period]) > 0 else 0
    
    # Wilder's smoothing
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period * 100 / atr[i] if atr[i] > 0 else 0
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period * 100 / atr[i] if atr[i] > 0 else 0
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) > 0 else 0
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period if i > period else 0
    
    # Load 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(19, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_down_aligned = align_htf_to_ltf(prices, df_1d, donchian_down)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, period+1)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up_aligned[i]) or 
            np.isnan(donchian_down_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_ratio = volume_4h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: Donchian breakout + volume confirmation + ADX filter
            # Long: break above upper band AND volume > 1.5x average AND ADX > 25
            if (close[i] > donchian_up_aligned[i] and 
                volume_ratio > 1.5 and
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: break below lower band AND volume > 1.5x average AND ADX > 25
            elif (close[i] < donchian_down_aligned[i] and 
                  volume_ratio > 1.5 and
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel
            if close[i] < donchian_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel
            if close[i] > donchian_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0