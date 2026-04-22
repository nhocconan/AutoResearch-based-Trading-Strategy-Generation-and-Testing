#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian Breakout with 1-week ADX trend filter and volume confirmation.
Trades breakouts of the 20-period Donchian channel in the direction of the weekly ADX trend (ADX>25).
Uses volume spike to confirm institutional interest at breakout. Designed for low trade frequency
(15-40 trades/year) to minimize fee decay and work in both bull and bear markets by requiring
strong trends (ADX filter) and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        plus_dm[i] = plus_dm[i] if plus_dm[i] > minus_dm[i] else 0
        minus_dm[i] = minus_dm[i] if minus_dm[i] > plus_dm[i] else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(high)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * np.where(atr != 0, 
                            np.convolve(plus_dm, np.ones(period)/period, mode='full')[period-1:-period+1] / atr, 0)
    minus_di = 100 * np.where(atr != 0,
                             np.convolve(minus_dm, np.ones(period)/period, mode='full')[period-1:-period+1] / atr, 0)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1)
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Donchian channel (20-period) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Strong trend filter: weekly ADX > 25
        strong_trend = adx_14_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and strong_trend and vol_spike:
            # Long: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low
            elif close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or trend weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or ADX weakens
                if close[i] < donchian_low[i] or adx_14_1w_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high or ADX weakens
                if close[i] > donchian_high[i] or adx_14_1w_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_ADX14_1w_Trend_Volume"
timeframe = "4h"
leverage = 1.0