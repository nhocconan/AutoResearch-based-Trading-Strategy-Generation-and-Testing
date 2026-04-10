#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX filter
# - Donchian(20) breakout from 4h: price > highest(high,20) for long, < lowest(low,20) for short
# - Volume confirmation: 4h volume > 2.0 x 20-period average (strong participation)
# - 1w ADX(14) > 25 to ensure trending market (avoid chop)
# - Stoploss: ATR-based trailing stop (exit when price moves against position by 2.5 * ATR)
# - Position size: 0.25 (25% of capital) to manage drawdown in bear markets
# - Works in bull/bear: ADX filter ensures we only trade when higher timeframe is trending
# - Target trades: ~150 over 4 years (37/year) to minimize fee drag

name = "4h_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 4h indicators
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 x 20-period average
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)
    
    # ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop long
    lowest_since_entry = 0.0   # for trailing stop short
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, high_4h[i])
            # Exit: price < lowest_low(20) OR trailing stop hit
            if close_4h[i] < lowest_low[i] or close_4h[i] < highest_since_entry - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, low_4h[i])
            # Exit: price > highest_high(20) OR trailing stop hit
            if close_4h[i] > highest_high[i] or close_4h[i] > lowest_since_entry + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout: price > highest_high(20) for long, < lowest_low(20) for short
                if close_4h[i] > highest_high[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_since_entry = high_4h[i]
                    signals[i] = 0.25
                elif close_4h[i] < lowest_low[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_since_entry = low_4h[i]
                    signals[i] = -0.25
    
    return signals