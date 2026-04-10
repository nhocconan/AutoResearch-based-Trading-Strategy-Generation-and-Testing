#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1h volume confirmation and 1d ADX trend filter
# - Long when price breaks above H3 (Camarilla resistance) AND 1h volume > 1.5x 20-period average AND 1d ADX > 25
# - Short when price breaks below L3 (Camarilla support) AND 1h volume > 1.5x 20-period average AND 1d ADX > 25
# - Exit when price crosses Camarilla pivot point (mean reversion)
# - Uses discrete position sizing 0.20 to minimize fee churn
# - Target: 15-35 trades/year on 1h (60-140 total over 4 years)
# - Works in bull/bear: volume confirms participation, daily ADX ensures we only trade when strong trend exists,
#   Camarilla levels provide structured support/resistance for breakouts

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Camarilla levels (based on previous day's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot and Camarilla levels for each 4h bar
    pivot_4h = np.full_like(high_4h, np.nan, dtype=float)
    h3_4h = np.full_like(high_4h, np.nan, dtype=float)
    l3_4h = np.full_like(high_4h, np.nan, dtype=float)
    
    for i in range(len(high_4h)):
        # Use previous 4h bar's OHLC (if available)
        if i > 0:
            phigh = high_4h[i-1]
            plow = low_4h[i-1]
            pclose = close_4h[i-1]
        else:
            # For first bar, use current bar (will be aligned later)
            phigh = high_4h[i]
            plow = low_4h[i]
            pclose = close_4h[i]
        
        pivot = (phigh + plow + pclose) / 3.0
        range_val = phigh - plow
        
        pivot_4h[i] = pivot
        h3_4h[i] = pivot + 1.1 * range_val / 2.0  # H3 resistance
        l3_4h[i] = pivot - 1.1 * range_val / 2.0  # L3 support
    
    # Pre-compute 1h volume average (20-period)
    volume_1h = prices['volume'].values
    vol_ma_1h = np.full_like(volume_1h, np.nan, dtype=float)
    for i in range(19, len(volume_1h)):
        vol_ma_1h[i] = np.mean(volume_1h[i-19:i+1])
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    # Align HTF indicators to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    vol_ma_1h_aligned = align_htf_to_ltf(prices, prices, vol_ma_1h)  # 1h data already in prices
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(vol_ma_1h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_price = prices['close'].values[i]
        volume_now = volume_1h[i]
        vol_ma_now = vol_ma_1h_aligned[i]
        
        # Volume spike condition (1.5x average)
        vol_spike = not np.isnan(vol_ma_now) and volume_now > 1.5 * vol_ma_now
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND 1d trending (ADX > 25)
            if (close_price > h3_4h_aligned[i] and vol_spike and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND volume spike AND 1d trending (ADX > 25)
            elif (close_price < l3_4h_aligned[i] and vol_spike and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses pivot point (mean reversion)
            exit_long = (position == 1 and close_price < pivot_4h_aligned[i])
            exit_short = (position == -1 and close_price > pivot_4h_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals