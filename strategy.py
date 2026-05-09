#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
# Long when: price breaks above Donchian upper, 12h ADX > 25 (trending), volume > 1.5x 20-period average.
# Short when: price breaks below Donchian lower, 12h ADX > 25, volume spike.
# Exit when: price crosses Donchian midpoint OR ADX < 20 (range).
# Position size: 0.25. Target: 20-40 trades/year to avoid fee drag.
# Works in bull (strong breakouts) and bear (strong breakdowns) via ADX filter.

name = "6h_Donchian20_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range (TR)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement (+DM, -DM)
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing, alpha = 1/14)
    def wilder_smooth(x, period):
        result = np.zeros_like(x)
        alpha = 1.0 / period
        result[period-1] = np.nansum(x[:period])  # seed
        for i in range(period, len(x)):
            result[i] = result[i-1] + alpha * (x[i] - result[i-1])
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # Trend conditions
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(adx_strong_aligned[i]) or
            np.isnan(adx_weak_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout up + strong trend + volume spike
            if (close[i] > donchian_upper[i] and 
                adx_strong_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown down + strong trend + volume spike
            elif (close[i] < donchian_lower[i] and 
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below midpoint OR trend weakens
            if (close[i] < donchian_mid[i]) or (adx_weak_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above midpoint OR trend weakens
            if (close[i] > donchian_mid[i]) or (adx_weak_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals