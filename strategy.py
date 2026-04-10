#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter
# - Long when price breaks above 20-period 4h Donchian high AND 12h volume > 1.5x 20-period average AND 1d ADX > 25
# - Short when price breaks below 20-period 4h Donchian low AND 12h volume > 1.5x 20-period average AND 1d ADX > 25
# - Exit when price returns to 4h Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 30-60 trades/year on 4h (120-240 total over 4 years)
# - Works in bull/bear: volume confirms breakout strength, daily ADX ensures we only trade in trending markets
# - 4h Donchian provides clear structure, 12h volume confirms participation, 1d ADX filters regime

name = "4h_12h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = np.full_like(high_4h, np.nan, dtype=float)
    donchian_low = np.full_like(low_4h, np.nan, dtype=float)
    donchian_mid = np.full_like(high_4h, np.nan, dtype=float)
    
    for i in range(19, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Pre-compute 12h volume average (20-period)
    volume_12h = df_12h['volume'].values
    vol_ma_12h = np.full_like(volume_12h, np.nan, dtype=float)
    for i in range(19, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
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
    
    # Align HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)  # 4h data already in prices
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 1.5 * vol_ma_4h[i]
        
        close_price = prices['close'].values[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above 4h Donchian high AND volume spike AND 1d trending (ADX > 25)
            if (close_price > donchian_high_aligned[i] and vol_spike and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: break below 4h Donchian low AND volume spike AND 1d trending (ADX > 25)
            elif (close_price < donchian_low_aligned[i] and vol_spike and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to 4h Donchian midpoint (mean reversion)
            exit_long = (position == 1 and close_price <= donchian_mid_aligned[i])
            exit_short = (position == -1 and close_price >= donchian_mid_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals