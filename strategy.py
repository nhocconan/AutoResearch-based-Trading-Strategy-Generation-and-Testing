#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 6h ADX trend filter
# - Long when price breaks above 12h Donchian upper channel (20-period) AND 1d volume > 1.5x 20-period average AND 6h ADX > 25
# - Short when price breaks below 12h Donchian lower channel (20-period) AND 1d volume > 1.5x 20-period average AND 6h ADX > 25
# - Exit when price returns to 12h Donchian midpoint (mean reversion within the channel)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 12-30 trades/year on 12h (50-120 total over 4 years)
# - Works in bull/bear: volume confirms participation, 6h ADX ensures we only trade when trend exists,
#   Donchian breakout captures momentum with defined risk

name = "12h_6h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian upper and lower channels
    donchian_upper = np.full_like(high_12h, np.nan, dtype=float)
    donchian_lower = np.full_like(low_12h, np.nan, dtype=float)
    donchian_mid = np.full_like(close_12h, np.nan, dtype=float)
    
    for i in range(19, len(high_12h)):
        donchian_upper[i] = np.max(high_12h[i-19:i+1])
        donchian_lower[i] = np.min(low_12h[i-19:i+1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 6h ADX(14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = np.abs(high_6h[1:] - low_6h[1:])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_6h[1:] - high_6h[:-1]) > (low_6h[:-1] - low_6h[1:]), 
                       np.maximum(high_6h[1:] - high_6h[:-1], 0), 0)
    dm_minus = np.where((low_6h[:-1] - low_6h[1:]) > (high_6h[1:] - high_6h[:-1]), 
                        np.maximum(low_6h[:-1] - low_6h[1:], 0), 0)
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
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_6h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_price = prices['close'].values[i]
        volume_now = prices['volume'].values[i]
        vol_spike = volume_now > 1.5 * vol_ma_1d_aligned[i]
        adx_trending = adx_6h_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian upper channel AND volume spike AND 6h trending (ADX > 25)
            if (close_price > donchian_upper[i] and vol_spike and adx_trending):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian lower channel AND volume spike AND 6h trending (ADX > 25)
            elif (close_price < donchian_lower[i] and vol_spike and adx_trending):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian midpoint (mean reversion within channel)
            exit_long = (position == 1 and close_price >= donchian_mid[i])
            exit_short = (position == -1 and close_price <= donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals