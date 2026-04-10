#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume spike + 1d ADX trend filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 1.5x 20-period average AND 1d ADX > 25
# - Short when price breaks below Donchian(20) low AND 12h volume > 1.5x 20-period average AND 1d ADX > 25
# - Exit when price crosses Donchian(10) midpoint (mean reversion within channel)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 25-40 trades/year on 4h (100-160 total over 4 years)
# - Works in bull/bear: volume confirms participation, daily ADX ensures we only trade when strong trend exists,
#   Donchian breakout captures momentum within the trend

name = "4h_12h_1d_donchian_breakout_volume_adx_v1"
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
    
    # Pre-compute 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) for breakout signals
    donchian_high_20 = np.full_like(high_4h, np.nan, dtype=float)
    donchian_low_20 = np.full_like(low_4h, np.nan, dtype=float)
    for i in range(19, len(high_4h)):
        donchian_high_20[i] = np.max(high_4h[i-19:i+1])
        donchian_low_20[i] = np.min(low_4h[i-19:i+1])
    
    # Donchian(10) for exit signals (midpoint)
    donchian_high_10 = np.full_like(high_4h, np.nan, dtype=float)
    donchian_low_10 = np.full_like(low_4h, np.nan, dtype=float)
    for i in range(9, len(high_4h)):
        donchian_high_10[i] = np.max(high_4h[i-9:i+1])
        donchian_low_10[i] = np.min(low_4h[i-9:i+1])
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
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
    donchian_high_20_aligned = align_htf_to_ltf(prices, prices, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, prices, donchian_low_20)
    donchian_mid_10_aligned = align_htf_to_ltf(prices, prices, donchian_mid_10)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_mid_10_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_price = prices['close'].values[i]
        volume_now = prices['volume'].values[i]
        vol_spike = volume_now > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian(20) high AND volume spike AND 1d ADX > 25
            if (close_price > donchian_high_20_aligned[i] and vol_spike and 
                adx_1d_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian(20) low AND volume spike AND 1d ADX > 25
            elif (close_price < donchian_low_20_aligned[i] and vol_spike and 
                  adx_1d_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian(10) midpoint (mean reversion)
            exit_long = (position == 1 and close_price < donchian_mid_10_aligned[i])
            exit_short = (position == -1 and close_price > donchian_mid_10_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals