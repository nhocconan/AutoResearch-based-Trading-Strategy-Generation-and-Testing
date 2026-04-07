#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily volume confirmation and ADX trend filter
# Donchian breakouts capture strong trending moves. Volume confirms institutional participation.
# ADX filter ensures we only trade in trending markets (ADX > 25), avoiding whipsaws in chop.
# Designed for low frequency in 12h timeframe (12-37 trades/year).
# Works in bull markets (breakouts to upside) and bear markets (breakouts to downside).

name = "12h_donchian20_volume_adx_v1"
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
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_1d - high_1d.shift(1)
    down_move = low_1d.shift(1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 12h Donchian(20) - using rolling window on 12h data
    # We need to calculate this on the 12h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trending = adx_12h_aligned[i] > 25
        
        # Exit conditions: exit when opposite breakout occurs or trend weakens
        if position == 1:  # Long position
            # Exit if breakdown occurs or ADX drops below 20 (trend weakening)
            if breakout_down or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if breakout occurs or ADX drops below 20 (trend weakening)
            if breakout_up or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: breakout up + volume confirmation + trending market
            if breakout_up and vol_confirm and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: breakdown down + volume confirmation + trending market
            elif breakout_down and vol_confirm and trending:
                position = -1
                signals[i] = -0.25
    
    return signals