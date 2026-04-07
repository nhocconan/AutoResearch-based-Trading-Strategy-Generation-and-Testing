#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour ADX trend strength filter with weekly Donchian breakout confirmation
# ADX(14) > 25 indicates strong trend, weekly Donchian(20) breakout provides entry in trend direction
# Volume confirmation ensures institutional participation. Designed for low frequency in 12h timeframe.
# Works in bull markets (trend continuation) and bear markets (strong downtrends).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_adx_weekly_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian breakout
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    donchian_high = high_1w.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate ADX on 12h data (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        if len(values) < period:
            return smoothed
        smoothed[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
        return smoothed
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(tr14 > 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 > 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend strength: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if trend weakens or price breaks below Donchian low
            if not strong_trend or close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if trend weakens or price breaks above Donchian high
            if not strong_trend or close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: strong trend + price above Donchian high + volume confirmation
            if strong_trend and close[i] > donchian_high_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: strong trend + price below Donchian low + volume confirmation
            elif strong_trend and close[i] < donchian_low_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals