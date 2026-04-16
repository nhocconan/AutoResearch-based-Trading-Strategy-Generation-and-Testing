#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R1/S1) breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above 12h Camarilla R1 AND 1d ADX > 20 AND 6h volume > 1.5x 20-period average.
# Short when price breaks below 12h Camarilla S1 AND 1d ADX > 20 AND 6h volume > 1.5x 20-period average.
# Exit when price crosses the 12h Camarilla pivot point (PP).
# Uses discrete position size 0.25. 12h/1d filters provide signal direction/regime, 6h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla formulas: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    camarilla_pp_12h = typical_price_12h
    camarilla_r1_12h = close_12h + range_12h * 1.1 / 12.0
    camarilla_s1_12h = close_12h - range_12h * 1.1 / 12.0
    
    # Shift by 1 to use previous bar's levels (no look-ahead)
    camarilla_pp_12h = np.roll(camarilla_pp_12h, 1)
    camarilla_r1_12h = np.roll(camarilla_r1_12h, 1)
    camarilla_s1_12h = np.roll(camarilla_s1_12h, 1)
    camarilla_pp_12h[0] = np.nan
    camarilla_r1_12h[0] = np.nan
    camarilla_s1_12h[0] = np.nan
    
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
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp_12h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        camarilla_pp = camarilla_pp_aligned[i]
        camarilla_r1 = camarilla_r1_aligned[i]
        camarilla_s1 = camarilla_s1_aligned[i]
        adx = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume average aligned
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < camarilla_pp:  # Exit when price crosses below pivot point
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > camarilla_pp:  # Exit when price crosses above pivot point
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND ADX > 20 AND volume > 1.5x 20-period avg
            if (price > camarilla_r1) and (adx > 20) and (vol > 1.5 * vol_ma_20_6h[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND ADX > 20 AND volume > 1.5x 20-period avg
            elif (price < camarilla_s1) and (adx > 20) and (vol > 1.5 * vol_ma_20_6h[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_12hCamarillaR1S1_1dADX_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0