#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period smoothed median, shifted 8), Teeth (8-period smoothed median, shifted 5), Lips (5-period smoothed median, shifted 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Exit when Alligator alignment reverses (Lips crosses Teeth or Jaw) OR price crosses Jaw
# - Alligator identifies trend absence (sleeping) vs trend formation (awakening) with convergence/divergence
# - 1w ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to avoid fee drag

name = "12h_1w_alligator_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h median price (typical price: (high+low+close)/3)
    median_price = (prices['high'] + prices['low'] + prices['close']) / 3
    median_price_values = median_price.values
    
    # Williams Alligator components on 12h
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_period = 13
    jaw_shift = 8
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_period = 8
    teeth_shift = 5
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_period = 5
    lips_shift = 3
    
    # Smoothed Moving Average (SMMA) - Wilder's smoothing (EMA with alpha=1/period)
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) >= period:
            # First value: simple average
            result[period-1] = np.mean(source[:period])
            # Wilder smoothing: SMMA[i] = (SMMA[i-1] * (period-1) + source[i]) / period
            for i in range(period, len(source)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaws_raw = smma(median_price_values, jaw_period)
    teeth_raw = smma(median_price_values, teeth_period)
    lips_raw = smma(median_price_values, lips_period)
    
    # Apply shifts (shift right = shift forward in time, so we use negative index offset)
    jaw = np.full_like(jaws_raw, np.nan, dtype=float)
    teeth = np.full_like(teeth_raw, np.nan, dtype=float)
    lips = np.full_like(lips_raw, np.nan, dtype=float)
    
    for i in range(len(jaws_raw)):
        if i - jaw_shift >= 0 and not np.isnan(jaws_raw[i]):
            jaw[i - jaw_shift] = jaws_raw[i]
        if i - teeth_shift >= 0 and not np.isnan(teeth_raw[i]):
            teeth[i - teeth_shift] = teeth_raw[i]
        if i - lips_shift >= 0 and not np.isnan(lips_raw[i]):
            lips[i - lips_shift] = lips_raw[i]
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder smoothing)
    tr_14 = np.full_like(tr, np.nan, dtype=float)
    dm_plus_14 = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14 = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(tr) >= 14:
        # Initial values (simple average)
        tr_14[13] = np.nanmean(tr[1:14])
        dm_plus_14[13] = np.nanmean(dm_plus[1:14])
        dm_minus_14[13] = np.nanmean(dm_minus[1:14])
        
        # Wilder smoothing
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14, np.nan, dtype=float)
    di_minus = np.full_like(tr_14, np.nan, dtype=float)
    mask = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[mask] = (dm_plus_14[mask] / tr_14[mask]) * 100
    di_minus[mask] = (dm_minus_14[mask] / tr_14[mask]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    mask_dx = (~np.isnan(di_plus) & ~np.isnan(di_minus) & 
               ((di_plus + di_minus) != 0))
    dx[mask_dx] = (np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / 
                   (di_plus[mask_dx] + di_minus[mask_dx])) * 100
    
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        # Initial ADX (simple average of first 14 DX)
        valid_dx = dx[14:28]  # indices 14 to 27
        if not np.all(np.isnan(valid_dx)):
            adx[27] = np.nanmean(valid_dx)
            # Wilder smoothing for ADX
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_lips = np.full(n, np.nan)  # for Lips-Teeth/Jaw cross detection
    prev_teeth = np.full(n, np.nan)
    prev_jaw = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after warmup
        # Store previous values for crossover detection
        if i > 0:
            prev_lips[i] = lips[i-1]
            prev_teeth[i] = teeth[i-1]
            prev_jaw[i] = jaw[i-1]
        else:
            prev_lips[i] = np.nan
            prev_teeth[i] = np.nan
            prev_jaw[i] = np.nan
        
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_12h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_12h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_12h[i]) and vol_series[i] > 1.5 * vol_ma_12h[i]
        
        close_price = prices['close'].values[i]
        lips_now = lips[i]
        teeth_now = teeth[i]
        jaw_now = jaw[i]
        lips_prev = prev_lips[i]
        teeth_prev = prev_teeth[i]
        jaw_prev = prev_jaw[i]
        
        # Alligator alignment signals
        bullish_alignment = lips_now > teeth_now > jaw_now
        bearish_alignment = lips_now < teeth_now < jaw_now
        
        # Lips crossing Teeth or Jaw (exit signals)
        lips_cross_teeth = (lips_prev <= teeth_prev and lips_now > teeth_now) or \
                          (lips_prev >= teeth_prev and lips_now < teeth_now)
        lips_cross_jaw = (lips_prev <= jaw_prev and lips_now > jaw_now) or \
                        (lips_prev >= jaw_prev and lips_now < jaw_now)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish alignment AND price > Lips AND 1w trending (ADX > 25) AND volume spike
            if (bullish_alignment and close_price > lips_now and 
                adx_1w_aligned[i] > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND price < Lips AND 1w trending (ADX > 25) AND volume spike
            elif (bearish_alignment and close_price < lips_now and 
                  adx_1w_aligned[i] > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Lips crosses Teeth OR Jaw OR alignment reverses
            exit_long = (position == 1 and 
                        (lips_cross_teeth or lips_cross_jaw or not bullish_alignment))
            exit_short = (position == -1 and 
                         (lips_cross_teeth or lips_cross_jaw or not bearish_alignment))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals