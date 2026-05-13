#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout
Hypothesis: Keltner Channel breakouts on 12h timeframe capture trend momentum, filtered by 1d ADX (trending market) and volume spike to avoid whipsaws. Long when price closes above upper KC (EMA20 + 2*ATR) with ADX>25 and volume>1.5x 20-period average. Short when price closes below lower KC (EMA20 - 2*ATR) with same filters. Uses 1d timeframe for trend filter to reduce noise and improve reliability in both bull and bear markets. Designed for low trade frequency (target 12-30/year) to minimize fee drag.
"""

name = "12h_Keltner_Channel_Breakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily timeframe (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values using Wilder's smoothing (EMA-like with alpha=1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period])
        plus_dm_sum = np.sum(plus_dm[1:period])
        minus_dm_sum = np.sum(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate EMA20 and ATR for Keltner Channel on 12h data
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    upper_kc = ema_20 + 2 * atr
    lower_kc = ema_20 - 2 * atr
    
    # Volume spike filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(upper_kc[i]) or np.isnan(lower_kc[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        # ADX trend filter: trending market (ADX > 25)
        trending = adx_14_aligned[i] > 25
        
        if position == 0:
            # LONG: Price closes above upper KC + volume spike + trending market
            if close[i] > upper_kc[i] and vol_spike and trending:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below lower KC + volume spike + trending market
            elif close[i] < lower_kc[i] and vol_spike and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 (middle of KC) or ADX weakens
            if close[i] < ema_20[i] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 or ADX weakens
            if close[i] > ema_20[i] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals