#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d ADX trend filter and volume confirmation
# - Long when price breaks above upper BB(20,2) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below lower BB(20,2) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses middle BB (20-period SMA) or opposite signal occurs
# - Bollinger Bands capture volatility expansion/contraction with clear breakout levels
# - 1d ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 20-50 trades/year on 4h (75-200 total over 4 years) to avoid fee drag

name = "4h_1d_bb_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute Bollinger Bands(20,2) on 4h
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    vol_4h = prices['volume'].values
    
    # Middle Band = SMA(20)
    sma_20 = np.full_like(close_4h, np.nan, dtype=float)
    for i in range(19, n):
        sma_20[i] = np.mean(close_4h[i-19:i+1])
    
    # Standard Deviation(20)
    std_20 = np.full_like(close_4h, np.nan, dtype=float)
    for i in range(19, n):
        std_20[i] = np.std(close_4h[i-19:i+1])
    
    # Upper Band = SMA + 2*Std
    upper_band = sma_20 + 2 * std_20
    # Lower Band = SMA - 2*Std
    lower_band = sma_20 - 2 * std_20
    
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
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_4h[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_4h[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_4h[i]
        upper_now = upper_band[i]
        lower_now = lower_band[i]
        middle_now = sma_20[i]
        adx_now = adx_1d_aligned[i]
        
        # Bollinger Band signals
        bb_break_up = close_now > upper_now  # price breaks above upper band
        bb_break_down = close_now < lower_now  # price breaks below lower band
        bb_cross_middle_up = (close_4h[i-1] <= middle_now and close_now > middle_now)  # crosses above middle
        bb_cross_middle_down = (close_4h[i-1] >= middle_now and close_now < middle_now)  # crosses below middle
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper BB AND 1d trending (ADX > 25) AND volume spike
            if (bb_break_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower BB AND 1d trending (ADX > 25) AND volume spike
            elif (bb_break_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses middle BB (mean reversion) or opposite BB breakout
            exit_long = (position == 1 and 
                        (bb_cross_middle_down or bb_break_down))
            exit_short = (position == -1 and 
                         (bb_cross_middle_up or bb_break_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals