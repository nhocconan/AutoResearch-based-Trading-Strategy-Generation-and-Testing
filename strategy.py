#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla H3 level (12h) AND 1d ADX > 25 AND 12h volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level (12h) AND 1d ADX > 25 AND 12h volume > 1.5x 20-period average
# - Exit when price returns to Camarilla Pivot level or opposite breakout occurs
# - Camarilla levels provide mathematically derived support/resistance with statistical edge
# - 1d ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to avoid fee drag

name = "12h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder smoothing
    tr_14 = np.full_like(tr, np.nan, dtype=float)
    dm_plus_14 = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14 = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(tr) >= 14:
        tr_14[13] = np.nanmean(tr[1:14])
        dm_plus_14[13] = np.nanmean(dm_plus[1:14])
        dm_minus_14[13] = np.nanmean(dm_minus[1:14])
        
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    di_plus = np.full_like(tr_14, np.nan, dtype=float)
    di_minus = np.full_like(tr_14, np.nan, dtype=float)
    mask = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[mask] = (dm_plus_14[mask] / tr_14[mask]) * 100
    di_minus[mask] = (dm_minus_14[mask] / tr_14[mask]) * 100
    
    dx = np.full_like(di_plus, np.nan, dtype=float)
    mask_dx = (~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0))
    dx[mask_dx] = (np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / (di_plus[mask_dx] + di_minus[mask_dx])) * 100
    
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        valid_dx = dx[14:28]
        if not np.all(np.isnan(valid_dx)):
            adx[27] = np.nanmean(valid_dx)
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Camarilla levels (based on previous day's range)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Camarilla levels (based on previous bar's range)
    camarilla_h5 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_h4 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_h3 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_h2 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_h1 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_l1 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_l2 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_l4 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_l5 = np.full_like(close_12h, np.nan, dtype=float)
    camarilla_pivot = np.full_like(close_12h, np.nan, dtype=float)
    
    for i in range(1, n):
        # Calculate based on previous bar's range
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        
        if np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
            continue
            
        range_val = prev_high - prev_low
        if range_val <= 0:
            continue
            
        camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3
        camarilla_h1[i] = camarilla_pivot[i] + (range_val * 1.0833 / 6)
        camarilla_h2[i] = camarilla_pivot[i] + (range_val * 1.0833 / 4)
        camarilla_h3[i] = camarilla_pivot[i] + (range_val * 1.0833 / 2)
        camarilla_h4[i] = camarilla_pivot[i] + (range_val * 1.0833)
        camarilla_h5[i] = camarilla_pivot[i] + (range_val * 1.0833 * 2)
        camarilla_l1[i] = camarilla_pivot[i] - (range_val * 1.0833 / 6)
        camarilla_l2[i] = camarilla_pivot[i] - (range_val * 1.0833 / 4)
        camarilla_l3[i] = camarilla_pivot[i] - (range_val * 1.0833 / 2)
        camarilla_l4[i] = camarilla_pivot[i] - (range_val * 1.0833)
        camarilla_l5[i] = camarilla_pivot[i] - (range_val * 1.0833 * 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_12h = prices['volume'].values
        vol_ma_20 = np.full_like(vol_12h, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_12h[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_12h[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_12h[i]
        camarilla_h3_now = camarilla_h3[i]
        camarilla_l3_now = camarilla_l3[i]
        camarilla_pivot_now = camarilla_pivot[i]
        adx_now = adx_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close_now > camarilla_h3_now  # price breaks above H3
        breakout_down = close_now < camarilla_l3_now  # price breaks below L3
        pivot_cross_up = (close_12h[i-1] <= camarilla_pivot_now and close_now > camarilla_pivot_now)  # crosses above pivot
        pivot_cross_down = (close_12h[i-1] >= camarilla_pivot_now and close_now < camarilla_pivot_now)  # crosses below pivot
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND 1d trending (ADX > 25) AND volume spike
            if (breakout_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND 1d trending (ADX > 25) AND volume spike
            elif (breakout_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot or opposite breakout
            exit_long = (position == 1 and 
                        (pivot_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (pivot_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals