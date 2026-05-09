#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 12h ADX > 25, volume spike (>1.5x 20-period average)
# Short when price breaks below Donchian(20) low, 12h ADX > 25, volume spike
# Exit when price crosses Donchian midline OR ADX drops below 20
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 25-50 trades/year.
# Designed to work in trending markets (bull/bear) with ADX filter to avoid whipsaws.

name = "4h_Donchian20_12hADX_Trend_Volume"
timeframe = "4h"
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
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h ADX(14) for trend strength
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    close_12h = df_12h['close']
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = abs(high_12h - close_12h.shift(1))
    tr3 = abs(low_12h - close_12h.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = high_12h - high_12h.shift(1)
    down = low_12h.shift(1) - low_12h
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothing
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_strong = adx_values > 25
    adx_weak = adx_values < 20
    
    # Align ADX signals to 4h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian high + ADX strong + volume spike
            if (close[i] > donchian_high[i] and 
                adx_strong_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian low + ADX strong + volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midline OR ADX weakens
            if (close[i] < donchian_mid[i]) or (adx_weak_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midline OR ADX weakens
            if (close[i] > donchian_mid[i]) or (adx_weak_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals