#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal with Volume Spike and 12h Trend Filter
# Camarilla pivot levels provide high-probability reversal zones in ranging markets
# Volume spike (2x average) confirms institutional interest at pivot levels
# 12h ADX < 20 ensures we only trade in ranging/low-volatility environments
# Works in bull markets (buy dips to support) and bear markets (sell rallies to resistance)
# Low turnover expected: ~15-30 trades/year per symbol

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX (14 periods) for trend strength
    adx_len = 14
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1-day Camarilla pivot levels
    # Using previous day's OHLC
    prev_day_open = np.roll(prices['open'].values, 1)
    prev_day_high = np.roll(prices['high'].values, 1)
    prev_day_low = np.roll(prices['low'].values, 1)
    prev_day_close = np.roll(prices['close'].values, 1)
    
    # First day has no previous data
    prev_day_open[0] = np.nan
    prev_day_high[0] = np.nan
    prev_day_low[0] = np.nan
    prev_day_close[0] = np.nan
    
    # Pivot point and support/resistance levels
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    range_hl = prev_day_high - prev_day_low
    
    # Camarilla levels
    r4 = pivot + range_hl * 1.1 / 2
    r3 = pivot + range_hl * 1.1 / 4
    r2 = pivot + range_hl * 1.1 / 6
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    s2 = pivot - range_hl * 1.1 / 6
    s3 = pivot - range_hl * 1.1 / 4
    s4 = pivot - range_hl * 1.1 / 2
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(pivot[i]) or 
            np.isnan(r1[i]) or 
            np.isnan(s1[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX < 20 indicates ranging market (good for mean reversion)
        ranging = adx_12h_aligned[i] < 20
        
        # Volume confirmation: current volume > 2x average
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: price at S1 support + volume spike + ranging market
            if (low[i] <= s1[i] and close[i] > s1[i] and 
                volume_spike and ranging):
                position = 1
                signals[i] = position_size
            # Enter short: price at R1 resistance + volume spike + ranging market
            elif (high[i] >= r1[i] and close[i] < r1[i] and 
                  volume_spike and ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot point or R1
            if close[i] >= pivot[i] or close[i] >= r1[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot point or S1
            if close[i] <= pivot[i] or close[i] <= s1[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Spike_Ranging_v1"
timeframe = "4h"
leverage = 1.0