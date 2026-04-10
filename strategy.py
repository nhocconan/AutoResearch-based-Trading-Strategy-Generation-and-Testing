#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Bull Power = High - EMA13(1d), Bear Power = EMA13(1d) - Low
# - Long when Bull Power > 0 AND Bear Power rising (improving momentum) AND 1d ADX > 20 (trending)
# - Short when Bear Power > 0 AND Bull Power falling (deteriorating momentum) AND 1d ADX > 20
# - Exit when power crosses zero or ADX < 15 (trend weakening)
# - Works in bull/bear: ADX ensures we only trade when trending, Elder Ray shows momentum shift
# - Discrete position sizing 0.25 to minimize fee churn
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years)

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d EMA13
    close_1d = df_1d['close'].values
    ema_13_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 13:
        # Simple average for first value
        ema_13_1d[12] = np.mean(close_1d[0:13])
        # EMA formula
        alpha = 2.0 / (13 + 1)
        for i in range(13, len(close_1d)):
            ema_13_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_13_1d[i-1]
    
    # Pre-compute 1d Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # High - EMA
    bear_power = ema_13_1d - low_1d   # EMA - Low
    
    # Pre-compute 1d ADX(14)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
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
        valid_dx = dx[14:28]
        if not np.all(np.isnan(valid_dx)):
            adx[27] = np.nanmean(valid_dx)
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (strong momentum) AND Bear Power rising (improving) AND trending
            bull_power_val = bull_power_aligned[i]
            bear_power_val = bear_power_aligned[i]
            bear_power_rising = (i > 100 and bear_power_val > bear_power_aligned[i-1])
            
            if (bull_power_val > 0 and bear_power_rising and 
                adx_1d_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power > 0 (strong momentum) AND Bull Power falling (deteriorating) AND trending
            elif (bear_power_val > 0 and 
                  bull_power_aligned[i] < bull_power_aligned[i-1] and 
                  adx_1d_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power crosses zero OR ADX < 15 (trend weakening)
            bull_power_val = bull_power_aligned[i]
            bear_power_val = bear_power_aligned[i]
            
            exit_long = (position == 1 and (bull_power_val <= 0 or adx_1d_aligned[i] < 15))
            exit_short = (position == -1 and (bear_power_val <= 0 or adx_1d_aligned[i] < 15))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals