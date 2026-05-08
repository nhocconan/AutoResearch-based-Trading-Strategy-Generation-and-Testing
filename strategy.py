#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 20-period high + volume > 1.5x average + ADX > 25.
# Short when price breaks below 20-period low + volume > 1.5x average + ADX > 25.
# Uses 1d ADX for trend strength filter to avoid whipsaws in ranging markets.
# Position size: 0.25 for clear trend, 0.125 for weakening trend.
# Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: ~25-50 trades per year to minimize fee drag while capturing strong moves.

name = "4h_Donchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_dc, lower_dc = donchian_channels(high, low, 20)
    
    # Volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20-1, len(volume)):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    volume_ratio = volume / vol_ma
    
    # ADX calculation (14-period) on 1d data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Smoothed values
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial averages
        atr[period] = np.mean(tr[1:period+1])
        plus_di[period] = np.mean(plus_dm[1:period+1]) / atr[period] * 100
        minus_di[period] = np.mean(minus_dm[1:period+1]) / atr[period] * 100
        
        # Smoothing
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100
        
        # DX and ADX
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume confirmation and trend strength
            if (close[i] > upper_dc[i] and 
                volume_ratio[i] > 1.5 and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lower_dc[i] and 
                  volume_ratio[i] > 1.5 and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below lower Donchian or weakening trend
            if close[i] < lower_dc[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above upper Donchian or weakening trend
            if close[i] > upper_dc[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals