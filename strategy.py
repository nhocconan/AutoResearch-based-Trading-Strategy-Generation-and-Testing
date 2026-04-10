#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h volume spike and 1d ADX trend filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 12h volume > 2.0x 20-period average AND 1d ADX > 20
# - Short when Williams %R(14) crosses below -20 (overbought) AND 12h volume > 2.0x 20-period average AND 1d ADX > 20
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 30-50 trades/year on 4h (120-200 total over 4 years)
# - Works in bull/bear: volume confirms participation, daily ADX ensures we only trade when trend exists,
#   Williams %R captures mean reversion within the trend

name = "4h_12h_1d_williamsr_volume_adx_v1"
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
    
    # Pre-compute 4h Williams %R(14)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    williams_r = np.full_like(close_4h, np.nan, dtype=float)
    for i in range(13, len(high_4h)):
        highest_high = np.max(high_4h[i-13:i+1])
        lowest_low = np.min(low_4h[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_4h[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # avoid division by zero
    
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
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # 4h data already in prices
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_williams_r = np.full(n, np.nan)  # for crossover detection
    
    for i in range(100, n):  # Start after warmup
        # Store previous Williams %R for crossover detection
        if i > 0:
            prev_williams_r[i] = williams_r_aligned[i-1]
        else:
            prev_williams_r[i] = np.nan
        
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(prev_williams_r[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (2.0x average)
        vol_series = prices['volume'].values
        vol_ma_4h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_4h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_4h[i]) and vol_series[i] > 2.0 * vol_ma_4h[i]
        
        close_price = prices['close'].values[i]
        williams_now = williams_r_aligned[i]
        williams_prev = prev_williams_r[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 (from below) AND volume spike AND 1d trending (ADX > 20)
            if (williams_prev <= -80 and williams_now > -80 and vol_spike and 
                adx_1d_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 (from above) AND volume spike AND 1d trending (ADX > 20)
            elif (williams_prev >= -20 and williams_now < -20 and vol_spike and 
                  adx_1d_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            exit_long = (position == 1 and williams_now >= -50)
            exit_short = (position == -1 and williams_now <= -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals