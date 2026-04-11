#!/usr/bin/env python3
# 4h_12h_donchian_breakout_volume_v3
# Strategy: 4h Donchian breakout with 12h volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum. Volume confirmation from 12h ensures institutional participation. ADX filter ensures we only trade in trending markets (ADX > 25), avoiding whipsaws in ranging markets. Works in both bull and bear by trading breakouts in the direction of the trend. Designed for low trade frequency (<50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Donchian channels on 4h (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    upper_band = highest_high.values
    lower_band = lowest_low.values
    
    # 12h ADX for trend strength (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.DataFrame(high_12h - low_12h)
    tr2 = pd.DataFrame(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.DataFrame(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_12h[0] - low_12h[0]  # First period
    
    # Directional Movement
    dm_plus = pd.DataFrame(np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                                    np.maximum(high_12h - np.roll(high_12h, 1), 0), 0))
    dm_minus = pd.DataFrame(np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                                     np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0))
    dm_plus.iloc[0] = 0
    dm_minus.iloc[0] = 0
    
    # Smoothed values
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus.values.flatten()).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus.values.flatten()).ewm(alpha=1/14, adjust=False).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr_12h
    di_minus = 100 * dm_minus_smooth / atr_12h
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_12h_values = adx_12h.values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_values)
    
    # 12h volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Current 12h volume aligned
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(adx_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current 12h volume > 20-period average
        vol_confirm = vol_12h_aligned[i] > vol_avg_20_12h_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above upper band AND strong trend AND volume confirmation
        if not np.isnan(upper_band[i]) and close[i] > upper_band[i] and strong_trend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below lower band AND strong trend AND volume confirmation
        elif not np.isnan(lower_band[i]) and close[i] < lower_band[i] and strong_trend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite band OR trend weakens
        elif position == 1 and (close[i] < lower_band[i] or adx_12h_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_band[i] or adx_12h_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals