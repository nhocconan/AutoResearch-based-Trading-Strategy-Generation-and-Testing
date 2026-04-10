#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversal with 1d ADX trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-20 to -80)
# - Long when Williams %R crosses above -80 (oversold) AND 1d ADX > 20 (trending) AND volume > 1.2x 20-period average
# - Short when Williams %R crosses below -20 (overbought) AND 1d ADX > 20 AND volume > 1.2x 20-period average
# - Exit when Williams %R crosses midpoint (-50) in opposite direction
# - Williams %R is effective in both trending and ranging markets when combined with trend filter
# - Target: 25-60 trades/year on 4h (100-240 total over 4 years) to balance opportunity and fee drag

name = "4h_1d_williamsr_meanrev_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Align HTF indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h Williams %R(14)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full_like(close_4h, np.nan, dtype=float)
    lowest_low = np.full_like(close_4h, np.nan, dtype=float)
    williams_r = np.full_like(close_4h, np.nan, dtype=float)
    
    for i in range(13, n):
        highest_high[i] = np.max(high_4h[i-13:i+1])
        lowest_low[i] = np.min(low_4h[i-13:i+1])
        hh_ll = highest_high[i] - lowest_low[i]
        if hh_ll != 0:
            williams_r[i] = ((highest_high[i] - close_4h[i]) / hh_ll) * -100
    
    # Pre-compute 4h volume MA(20) for volume confirmation
    vol_4h = prices['volume'].values
    vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(vol_4h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation (1.2x average)
        vol_spike = vol_4h[i] > 1.2 * vol_ma_20[i]
        
        williams_now = williams_r[i]
        williams_prev = williams_r[i-1] if i > 0 else williams_r[i]
        adx_now = adx_1d_aligned[i]
        
        # Williams %R signals
        williams_oversold = williams_now < -80  # Oversold
        williams_overbought = williams_now > -20  # Overbought
        williams_cross_above_80 = (williams_prev <= -80 and williams_now > -80)  # Cross above -80
        williams_cross_below_20 = (williams_prev >= -20 and williams_now < -20)  # Cross below -20
        williams_cross_above_50 = (williams_prev <= -50 and williams_now > -50)  # Cross above -50
        williams_cross_below_50 = (williams_prev >= -50 and williams_now < -50)  # Cross below -50
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 (from oversold) AND 1d trending (ADX > 20) AND volume confirmation
            if (williams_cross_above_80 and adx_now > 20 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 (from overbought) AND 1d trending (ADX > 20) AND volume confirmation
            elif (williams_cross_below_20 and adx_now > 20 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses midpoint (-50) in opposite direction
            exit_long = (position == 1 and williams_cross_below_50)
            exit_short = (position == -1 and williams_cross_above_50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals