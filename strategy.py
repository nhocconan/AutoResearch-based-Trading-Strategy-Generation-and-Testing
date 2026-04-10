#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ADX trend filter
# - Long when price breaks above Donchian upper band (20-period high) AND 12h ADX > 25 AND volume > 2.0x 20-period average
# - Short when price breaks below Donchian lower band (20-period low) AND 12h ADX > 25 AND volume > 2.0x 20-period average
# - Exit when price crosses the Donchian middle (10-period average of upper/lower) or opposite signal occurs
# - Donchian channels provide clear structure for breakouts in both bull and bear markets
# - 12h ADX filter ensures we only trade when higher timeframe is strongly trending (avoids chop)
# - Volume confirmation prevents false signals in low participation
# - Target: 19-50 trades/year on 4h (75-200 total over 4 years) to avoid fee drag

name = "4h_12h_donchian_breakout_volume_adx_v1"
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
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper band (20-period high)
    donchian_upper = np.full_like(high_4h, np.nan, dtype=float)
    for i in range(19, len(high_4h)):
        donchian_upper[i] = np.max(high_4h[i-19:i+1])
    
    # Donchian lower band (20-period low)
    donchian_lower = np.full_like(low_4h, np.nan, dtype=float)
    for i in range(19, len(low_4h)):
        donchian_lower[i] = np.min(low_4h[i-19:i+1])
    
    # Donchian middle band (10-period average of upper/lower)
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (2.0x average)
        vol_4h = prices['volume'].values
        vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_4h[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_4h[i] > 2.0 * vol_ma_20[i]
        
        close_now = close_4h[i]
        high_now = high_4h[i]
        low_now = low_4h[i]
        donchian_upper_now = donchian_upper[i]
        donchian_lower_now = donchian_lower[i]
        donchian_middle_now = donchian_middle[i]
        adx_now = adx_12h_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close_now > donchian_upper_now  # price breaks above upper band
        breakout_down = close_now < donchian_lower_now  # price breaks below lower band
        middle_cross_up = (close_4h[i-1] <= donchian_middle_now and close_now > donchian_middle_now)  # crosses above middle
        middle_cross_down = (close_4h[i-1] >= donchian_middle_now and close_now < donchian_middle_now)  # crosses below middle
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper band AND 12h trending (ADX > 25) AND volume spike
            if (breakout_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower band AND 12h trending (ADX > 25) AND volume spike
            elif (breakout_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses middle band (mean reversion) or opposite Donchian breakout
            exit_long = (position == 1 and 
                        (middle_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (middle_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals