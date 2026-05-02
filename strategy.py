#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13, ADX(14) > 25 filters for trending markets
# Volume confirmation ensures breakout validity. Works in bull markets (bull power > 0 + ADX up) 
# and bear markets (bear power < 0 + ADX up). Targets 12-37 trades/year via strict confluence.

name = "6h_ElderRay_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX regime filter and EMA13 (for Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX/EMA calculation
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX components (DI+, DI-, DX) for trend strength
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift(1))).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = -pd.Series(df_1d['low']).diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = atr  # Already smoothed
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Bull Power and Bear Power (Elder Ray)
    bull_power = (pd.Series(df_1d['high']) - ema_13_1d).values
    bear_power = (ema_13_1d - pd.Series(df_1d['low'])).values
    
    # Align all 1d indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull power positive + trending + volume confirmation
            if bull_power_aligned[i] > 0 and trending and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive + trending + volume confirmation
            elif bear_power_aligned[i] > 0 and trending and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull power turns negative OR trend weakens (ADX < 20)
            if bull_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear power turns negative OR trend weakens (ADX < 20)
            if bear_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals