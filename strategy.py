#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ATR filter and volume confirmation.
# Breakouts from Donchian channels capture strong momentum moves. The 1-day ATR filter
# ensures sufficient volatility, while volume confirmation validates the breakout.
# This combination works in both bull and bear markets by catching strong directional moves.
# Target: 15-30 trades/year (60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Donchian and ATR ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) on daily data
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(len(high)):
            if i >= period - 1:
                upper[i] = np.max(high[i - period + 1:i + 1])
                lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    donchian_upper_1d, donchian_lower_1d = calculate_donchian(high_1d, low_1d, 20)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # ATR calculation on daily data (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d data for volume average ===
    volume_1d = df_1d['volume'].values
    vol_avg20_1d = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_avg20_1d[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_avg20_1d[i] = np.nan
    
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = 100  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + sufficient volatility + volume
            if close[i] > donchian_upper_aligned[i] and \
               atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + sufficient volatility + volume
            elif close[i] < donchian_lower_aligned[i] and \
                 atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0