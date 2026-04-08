#!/usr/bin/env python3
"""
12h_donchian_breakout_volume_v1
Hypothesis: Breakouts above/below Donchian channels with volume confirmation and ADX trend filter.
- Long: Price breaks above Donchian upper channel + volume > 1.5x avg + ADX > 25
- Short: Price breaks below Donchian lower channel + volume > 1.5x avg + ADX > 25
- Exit: Price crosses opposite Donchian channel
- Uses 1d trend filter: only trade long when price > 1d EMA50, short when price < 1d EMA50
- Target: 15-30 trades/year to avoid overtrading on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_v1"
timeframe = "12h"
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
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= lookback:
            vol_sum -= volume[i - lookback]
        if i >= lookback - 1:
            vol_ma[i] = vol_sum / lookback
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
        
        # Smooth TR, +DM, -DM
        atr = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        # Initial average
        if n >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_avg = np.mean(plus_dm[1:period])
            minus_dm_avg = np.mean(minus_dm[1:period])
            
            if atr[period-1] != 0:
                plus_di[period-1] = (plus_dm_avg / atr[period-1]) * 100
                minus_di[period-1] = (minus_dm_avg / atr[period-1]) * 100
            
            # Wilder smoothing
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_avg = (plus_dm_avg * (period-1) + plus_dm[i]) / period
                minus_dm_avg = (minus_dm_avg * (period-1) + minus_dm[i]) / period
                
                if atr[i] != 0:
                    plus_di[i] = (plus_dm_avg / atr[i]) * 100
                    minus_di[i] = (minus_dm_avg / atr[i]) * 100
        
        # Calculate DX and ADX
        dx = np.zeros(n)
        adx = np.zeros(n)
        
        for i in range(period, n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        # Smooth DX to get ADX
        if n >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    
    # Calculate EMA50
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50[i] = close_1d[i]
        elif np.isnan(ema_50[i-1]):
            ema_50[i] = close_1d[i]
        else:
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price crosses below lower Donchian channel
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above upper Donchian channel
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long conditions
            long_breakout = close[i] > highest_high[i]
            long_volume = volume[i] > 1.5 * vol_ma[i]
            long_adx = adx[i] > 25
            long_trend = close[i] > ema_50_aligned[i]
            
            if long_breakout and long_volume and long_adx and long_trend:
                position = 1
                signals[i] = 0.25
            # Short conditions
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  adx[i] > 25 and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals