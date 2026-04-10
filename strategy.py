#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Camarilla levels from 1d: L3, H3, L4, H4 as key support/resistance
# - Breakout above H4 with volume = long, breakdown below L4 with volume = short
# - 1d ADX(14) > 25 to ensure trending market and avoid chop
# - Volume confirmation: current 4h volume > 1.8x 20-period average
# - ATR-based trailing stop: exit long when price < highest_high - 2.5*ATR, exit short when price > lowest_low + 2.5*ATR
# - Designed for 4h timeframe: targets 20-50 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_camarilla_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1d Camarilla levels
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = high_1d[0]
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    camarilla_range = high_1d_prev - low_1d_prev
    camarilla_H4 = close_1d_prev + camarilla_range * 1.1 / 2
    camarilla_L4 = close_1d_prev - camarilla_range * 1.1 / 2
    camarilla_H3 = close_1d_prev + camarilla_range * 1.1 / 4
    camarilla_L3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h
    H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Pre-compute 4h Donchian channels (20-period) for breakout confirmation
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper band: highest high over past 20 periods
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over past 20 periods
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.8 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_4h[i] > highest_high:
                highest_high = close_4h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_4h[i] < highest_high - 2.5 * atr_14[i] or close_4h[i] < highest_20[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_4h[i] < lowest_low:
                lowest_low = close_4h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_4h[i] > lowest_low + 2.5 * atr_14[i] or close_4h[i] > lowest_20[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above H4 and above Donchian upper band
                if close_4h[i] > H4_aligned[i] and close_4h[i] > highest_20[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_high = close_4h[i]
                    signals[i] = 0.25
                # Breakout short: price closes below L4 and below Donchian lower band
                elif close_4h[i] < L4_aligned[i] and close_4h[i] < lowest_20[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_low = close_4h[i]
                    signals[i] = -0.25
    
    return signals