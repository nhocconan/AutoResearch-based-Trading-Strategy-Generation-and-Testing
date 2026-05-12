#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h ADX trend filter and volume spike.
Works in bull/bear markets because: 1) Donchian channels capture breakouts in any market regime, 2) 4h ADX > 25 ensures we only trade strong trends, avoiding whipsaws in ranges, 3) Volume spike confirms breakout validity, reducing false signals. Target 20-40 trades/year.
"""
name = "1h_Donchian20_Breakout_4hADX_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h DATA FOR ADX TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = high[i] - high[i-1] if (high[i] - high[i-1]) > (low[i-1] - low[i]) and (high[i] - high[i-1]) > 0 else 0
            minus_dm[i] = low[i-1] - low[i] if (low[i-1] - low[i]) > (high[i] - high[i-1]) and (low[i-1] - low[i]) > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smooth(data, period):
            smoothed = np.zeros_like(data)
            smoothed[period-1] = np.nansum(data[:period])  # Simple average for first value
            for i in range(period, len(data)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + (data[i] / period)
            return smoothed
        
        if len(tr) < period:
            return np.full_like(high, np.nan)
        
        atr = wilders_smooth(tr, period)
        plus_di = 100 * wilders_smooth(plus_dm, period) / atr
        minus_di = 100 * wilders_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # === 1h DONCHIAN CHANNEL (20-period) ===
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # 20 for Donchian, 34 for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian, ADX > 25 (strong trend), volume spike
            if (close[i] > upper[i] and 
                adx_4h_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below lower Donchian, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower[i] and 
                  adx_4h_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or ADX weakens (< 20)
            if (close[i] < lower[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or ADX weakens (< 20)
            if (close[i] > upper[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals