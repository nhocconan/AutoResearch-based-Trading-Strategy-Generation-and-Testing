#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume confirmation and 1w trend filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 1d ATR (volatility-adjusted volume spike) AND 1w ADX > 20 (weak trend filter)
# - Short when price breaks below Donchian(20) low AND same conditions
# - Exit when price crosses Donchian midpoint (mean reversion)
# - Discrete position sizing 0.25 to minimize fee churn
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years)
# - Works in bull/bear: volume confirmation ensures breakout strength, weekly ADX avoids choppy markets

name = "4h_1d_1w_donchian_atr_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian(20) levels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    vol_4h = prices['volume'].values
    
    # Donchian channels
    donch_high = np.full_like(high_4h, np.nan, dtype=float)
    donch_low = np.full_like(low_4h, np.nan, dtype=float)
    donch_mid = np.full_like(close_4h, np.nan, dtype=float)
    
    for i in range(19, len(high_4h)):
        donch_high[i] = np.max(high_4h[i-19:i+1])
        donch_low[i] = np.min(low_4h[i-19:i+1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Pre-compute 1d ATR(14) for volume normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) - Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        # Initial ATR (simple average)
        atr_14[13] = np.nanmean(tr[1:14])
        # Wilder smoothing
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume / ATR ratio (volatility-adjusted volume)
    volume_1d = df_1d['volume'].values
    vol_atr_ratio = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(len(volume_1d)):
        if not np.isnan(volume_1d[i]) and not np.isnan(atr_14[i]) and atr_14[i] > 0:
            vol_atr_ratio[i] = volume_1d[i] / atr_14[i]
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1w = np.abs(high_1w[1:] - low_1w[1:])
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    trw = np.concatenate([[np.nan], trw])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14w = np.full_like(trw, np.nan, dtype=float)
    dm_plus_14w = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14w = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(trw) >= 14:
        # Initial values (simple average)
        tr_14w[13] = np.nanmean(trw[1:14])
        dm_plus_14w[13] = np.nanmean(dm_plus[1:14])
        dm_minus_14w[13] = np.nanmean(dm_minus[1:14])
        
        # Wilder smoothing
        for i in range(14, len(trw)):
            tr_14w[i] = (tr_14w[i-1] * 13 + trw[i]) / 14
            dm_plus_14w[i] = (dm_plus_14w[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_14w[i] = (dm_minus_14w[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14w, np.nan, dtype=float)
    di_minus = np.full_like(tr_14w, np.nan, dtype=float)
    mask = ~np.isnan(tr_14w) & (tr_14w != 0)
    di_plus[mask] = (dm_plus_14w[mask] / tr_14w[mask]) * 100
    di_minus[mask] = (dm_minus_14w[mask] / tr_14w[mask]) * 100
    
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
    donch_high_aligned = align_htf_to_ltf(prices, prices, donch_high)  # same timeframe
    donch_low_aligned = align_htf_to_ltf(prices, prices, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, prices, donch_mid)
    vol_atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_atr_ratio)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_atr_ratio_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (2.0x ATR-adjusted volume - more selective than raw volume)
        vol_spike = vol_atr_ratio_aligned[i] > 2.0
        
        close_price = prices['close'].values[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: break above Donchian high AND volume spike AND 1w trending (ADX > 20)
            if (close_price > donch_high_aligned[i] and vol_spike and 
                adx_1w_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short conditions: break below Donchian low AND volume spike AND 1w trending (ADX > 20)
            elif (close_price < donch_low_aligned[i] and vol_spike and 
                  adx_1w_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midpoint (mean reversion)
            exit_long = (position == 1 and close_price <= donch_mid_aligned[i])
            exit_short = (position == -1 and close_price >= donch_mid_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals