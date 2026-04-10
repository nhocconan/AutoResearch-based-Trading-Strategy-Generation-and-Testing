#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# - Long when price breaks above Donchian upper band (20) AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian lower band (20) AND 1w ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses Donchian midpoint or opposite signal occurs
# - Donchian channels provide clear trend structure with proven efficacy on SOLUSDT
# - 1w ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 7-25 trades/year on 1d (30-100 total over 4 years) to avoid fee drag

name = "1d_1w_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
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
    
    # Align HTF indicators to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    donchian_upper = np.full_like(close_1d, np.nan, dtype=float)
    donchian_lower = np.full_like(close_1d, np.nan, dtype=float)
    donchian_mid = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(19, n):  # Need 20 periods for Donchian
        donchian_upper[i] = np.max(high_1d[i-19:i+1])
        donchian_lower[i] = np.min(low_1d[i-19:i+1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_1d = prices['volume'].values
        vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_1d[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_1d[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_1d[i]
        donchian_upper_now = donchian_upper[i]
        donchian_lower_now = donchian_lower[i]
        donchian_mid_now = donchian_mid[i]
        adx_now = adx_1w_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close_now > donchian_upper_now  # price breaks above upper band
        breakout_down = close_now < donchian_lower_now  # price breaks below lower band
        midpoint_cross_up = (close_1d[i-1] <= donchian_mid_now and close_now > donchian_mid_now)  # crosses above midpoint
        midpoint_cross_down = (close_1d[i-1] >= donchian_mid_now and close_now < donchian_mid_now)  # crosses below midpoint
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper band AND 1w trending (ADX > 25) AND volume spike
            if (breakout_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower band AND 1w trending (ADX > 25) AND volume spike
            elif (breakout_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses midpoint or opposite Donchian breakout
            exit_long = (position == 1 and 
                        (midpoint_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (midpoint_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals