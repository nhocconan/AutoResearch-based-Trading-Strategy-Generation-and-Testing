#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels (R4/S4) for breakout/continuation and 1d ADX(14) for trend filter.
# Long when price breaks above 1w Camarilla R4 AND 1d ADX > 20 AND 6h volume > 1.3x 20-period average.
# Short when price breaks below 1w Camarilla S4 AND 1d ADX > 20 AND 6h volume > 1.3x 20-period average.
# Exit when price crosses the 1w Camarilla pivot point (PP).
# Uses discrete position size 0.25. 1w/1d filters provide signal direction, 6h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w Indicators: Camarilla Pivots (based on prior week) ===
    # Camarilla levels: PP = (H+L+C)/3, R4 = PP + 1.1*(H-L)*1.1/2, S4 = PP - 1.1*(H-L)*1.1/2
    # Note: Using prior week's OHLC to avoid look-ahead
    pp_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r4_1w = pp_1w + 1.1 * (np.roll(high_1w, 1) - np.roll(low_1w, 1)) * 1.1 / 2.0
    s4_1w = pp_1w - 1.1 * (np.roll(high_1w, 1) - np.roll(low_1w, 1)) * 1.1 / 2.0
    
    # Set first value to NaN (no prior week)
    pp_1w[0] = np.nan
    r4_1w[0] = np.nan
    s4_1w[0] = np.nan
    
    # === 1d Indicators: ADX (14) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_1d = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_1d = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14_1d = 100 * dm_plus_14_1d / atr_14_1d
    di_minus_14_1d = 100 * dm_minus_14_1d / atr_14_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_14_1d - di_minus_14_1d) / (di_plus_14_1d + di_minus_14_1d)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_14_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        pp = pp_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        adx = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume average aligned
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < pp:  # Exit when price crosses below pivot point
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > pp:  # Exit when price crosses above pivot point
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND ADX > 20 AND volume > 1.3x 20-period avg
            if (price > r4) and (adx > 20) and (vol > 1.3 * vol_ma_20_6h[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S4 AND ADX > 20 AND volume > 1.3x 20-period avg
            elif (price < s4) and (adx > 20) and (vol > 1.3 * vol_ma_20_6h[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1wCamarillaR4S4_1dADX_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0