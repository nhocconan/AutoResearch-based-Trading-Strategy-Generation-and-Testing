#!/usr/bin/env python3
# 1d_1w_price_channel_volume_sr_v1
# Hypothesis: 1d price channel (Donchian) breakout with volume confirmation and 1w trend filter.
# Long on upper band breakout with volume surge in uptrend; short on lower band breakout with volume surge in downtrend.
# Uses 1w ADX to filter weak trends. Designed for 15-25 trades/year on 1d to avoid fee drag.
# Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_price_channel_volume_sr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest[i] = np.max(high[i - period + 1:i + 1])
        lowest[i] = np.min(low[i - period + 1:i + 1])
    
    # Volume moving average (20-period) for surge detection
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= period:
            vol_sum -= volume[i - period]
        if i >= period - 1:
            vol_ma[i] = vol_sum / period
    
    # Volume surge: current volume > 1.5 * 20-period average
    vol_surge = volume > (vol_ma * 1.5)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.full(n, np.nan)
        dm_plus = np.full(n, np.nan)
        dm_minus = np.full(n, np.nan)
        
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            
            if high_diff > low_diff and high_diff > 0:
                dm_plus[i] = high_diff
            else:
                dm_plus[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                dm_minus[i] = low_diff
            else:
                dm_minus[i] = 0
        
        # Smoothed values
        atr = np.full(n, np.nan)
        signal_plus = np.full(n, np.nan)
        signal_minus = np.full(n, np.nan)
        
        # Initial values
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            signal_plus[period-1] = np.nanmean(dm_plus[1:period])
            signal_minus[period-1] = np.nanmean(dm_minus[1:period])
            
            # Wilder smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                signal_plus[i] = (signal_plus[i-1] * (period - 1) + dm_plus[i]) / period
                signal_minus[i] = (signal_minus[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Avoid division by zero
        dx = np.full(n, np.nan)
        for i in range(period, n):
            if atr[i] != 0:
                dx[i] = (abs(signal_plus[i] - signal_minus[i]) / (signal_plus[i] + signal_minus[i])) * 100
        
        adx = np.full(n, np.nan)
        for i in range(2*period - 1, n):
            if i == 2*period - 1:
                adx[i] = np.nanmean(dx[period:i+1])
            else:
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx_1w_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period - 1, 2*14 - 1)  # Donchian and ADX ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band or trend weakens
            if close[i] < lowest[i] or not strong_trend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band or trend weakens
            if close[i] > highest[i] or not strong_trend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume surge in strong uptrend
            if (close[i] > highest[i] and 
                vol_surge[i] and 
                strong_trend[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume surge in strong downtrend
            elif (close[i] < lowest[i] and 
                  vol_surge[i] and 
                  strong_trend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals