#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian channel breakout (20) with volume confirmation and ADX trend filter.
# Donchian breakouts capture trend continuation; volume confirms institutional participation; ADX > 25 filters for trending markets.
# Designed for low trade frequency (20-30/year) to minimize fee drift while capturing strong trends in both bull and bear markets.
# Uses 1d timeframe for ADX calculation to avoid whipsaw on lower timeframes.

name = "4h_Donchian20_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i]) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian channel (20) on 4h
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above upper Donchian with volume and ADX > 25
            if close[i] > dc_upper[i] and vol_confirm[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below lower Donchian with volume and ADX > 25
            elif close[i] < dc_lower[i] and vol_confirm[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below lower Donchian or ADX < 20 (trend weakening)
            if close[i] < dc_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above upper Donchian or ADX < 20 (trend weakening)
            if close[i] > dc_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals