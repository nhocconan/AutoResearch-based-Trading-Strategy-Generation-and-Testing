#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla H3 level AND 12h ADX > 20 AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level AND 12h ADX > 20 AND volume > 1.5x 20-period average
# - Exit when price crosses the Camarilla Pivot (midpoint) or opposite signal occurs
# - Camarilla levels provide intraday support/resistance that works in both trending and ranging markets
# - 12h ADX filter ensures we trade when higher timeframe has directional bias
# - Volume confirmation prevents false breakouts in low liquidity
# - Target: 25-40 trades/year on 4h (100-160 total over 4 years) to avoid fee drag

name = "4h_12h_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
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
    
    # Align HTF indicators to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 4h Camarilla levels (based on previous day's OHLC)
    # We'll use daily OHLC to compute Camarilla levels for the 4h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_h3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_h2 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_h1 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_pivot = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l1 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l2 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_l4 = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(len(high_1d)):
        if i == 0:
            continue  # Skip first day as we need previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        camarilla_pivot[i] = pivot
        camarilla_h1[i] = pivot + (range_val * 1.1 / 12)
        camarilla_h2[i] = pivot + (range_val * 1.1 / 6)
        camarilla_h3[i] = pivot + (range_val * 1.1 / 4)
        camarilla_h4[i] = pivot + (range_val * 1.1 / 2)
        camarilla_l1[i] = pivot - (range_val * 1.1 / 12)
        camarilla_l2[i] = pivot - (range_val * 1.1 / 6)
        camarilla_l3[i] = pivot - (range_val * 1.1 / 4)
        camarilla_l4[i] = pivot - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Pre-compute 4h price arrays
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    vol_4h = prices['volume'].values
    
    # Pre-compute 20-period volume average
    vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
    for i in range(19, len(vol_4h)):
        vol_ma_20[i] = np.mean(vol_4h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_1d[i]) or np.isnan(camarilla_l3_1d[i]) or 
            np.isnan(camarilla_pivot_1d[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_spike = vol_4h[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_4h[i]
        high_now = high_4h[i]
        low_now = low_4h[i]
        camarilla_h3_now = camarilla_h3_1d[i]
        camarilla_l3_now = camarilla_l3_1d[i]
        camarilla_pivot_now = camarilla_pivot_1d[i]
        adx_now = adx_12h_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close_now > camarilla_h3_now  # price breaks above H3
        breakout_down = close_now < camarilla_l3_now  # price breaks below L3
        pivot_cross_up = (close_4h[i-1] <= camarilla_pivot_now and close_now > camarilla_pivot_now)  # crosses above pivot
        pivot_cross_down = (close_4h[i-1] >= camarilla_pivot_now and close_now < camarilla_pivot_now)  # crosses below pivot
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 12h trending (ADX > 20) AND volume spike
            if (breakout_up and adx_now > 20 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND 12h trending (ADX > 20) AND volume spike
            elif (breakout_down and adx_now > 20 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses pivot (mean reversion) or opposite Camarilla breakout
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