#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter.
Long when price breaks above Donchian upper band AND 1d volume > 2x 20-period average AND 1w ADX > 20.
Short when price breaks below Donchian lower band AND 1d volume > 2x 20-period average AND 1w ADX > 20.
Exit on opposite Donchian band touch or 1w ADX < 15 (trend weakening).
Donchian channels provide robust trend-following structure proven on SOLUSDT.
1d volume spike confirms breakout legitimacy. 1w ADX > 20 ensures we only trade strong weekly trends.
Designed for 4h timeframe targeting 75-200 total trades over 4 years with moderate frequency.
Works in both bull and bear markets by only taking breakouts in direction of strong weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume spike filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d 20-period volume average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 1w data
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
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (np.sum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.sum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w)
    
    # Align 1d volume average and 1w ADX to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dc_upper = donchian_upper[i]
        dc_lower = donchian_lower[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        adx_val = adx_1w_aligned[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 1d volume spike AND 1w ADX > 20 (strong trend)
            if (price > dc_upper and volume[i] > 2.0 * vol_ma_val and adx_val > 20):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND 1d volume spike AND 1w ADX > 20 (strong trend)
            elif (price < dc_lower and volume[i] > 2.0 * vol_ma_val and adx_val > 20):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches Donchian lower OR 1w ADX < 15 (trend weakening)
                if (price <= dc_lower or adx_val < 15):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches Donchian upper OR 1w ADX < 15 (trend weakening)
                if (price >= dc_upper or adx_val < 15):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dVolumeSpike_1wADX_Trend"
timeframe = "4h"
leverage = 1.0