#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w ADX trend filter and volume confirmation
# - Williams %R(14): momentum oscillator measuring overbought/oversold levels
# - Long when %R crosses above -80 from below AND 1w ADX > 25 AND volume > 1.3x 20-period average
# - Short when %R crosses below -20 from above AND 1w ADX > 25 AND volume > 1.3x 20-period average
# - Exit when %R crosses -50 (mean reversion) or opposite signal occurs
# - Williams %R identifies turning points in trending markets with less lag than RSI
# - 1w ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to avoid fee drag

name = "12h_1w_williamsr_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 12h
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full_like(close_12h, np.nan, dtype=float)
    lookback = 14
    
    for i in range(lookback - 1, n):
        highest_high = np.max(high_12h[i-lookback+1:i+1])
        lowest_low = np.min(low_12h[i-lookback+1:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close_12h[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
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
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_williams_r = np.full(n, np.nan)  # for crossover detection
    
    for i in range(100, n):  # Start after warmup
        # Store previous value for crossover detection
        if i > 0:
            prev_williams_r[i] = williams_r[i-1]
        else:
            prev_williams_r[i] = np.nan
        
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.3x average)
        vol_series = prices['volume'].values
        vol_ma_12h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_12h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_12h[i]) and vol_series[i] > 1.3 * vol_ma_12h[i]
        
        williams_now = williams_r[i]
        williams_prev = prev_williams_r[i]
        adx_now = adx_1w_aligned[i]
        
        # Williams %R signals
        williams_cross_80_up = (williams_prev <= -80 and williams_now > -80)  # crosses above -80
        williams_cross_20_down = (williams_prev >= -20 and williams_now < -20)  # crosses below -20
        williams_cross_50_up = (williams_prev <= -50 and williams_now > -50)  # crosses above -50
        williams_cross_50_down = (williams_prev >= -50 and williams_now < -50)  # crosses below -50
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: %R crosses above -80 AND 1w trending (ADX > 25) AND volume spike
            if (williams_cross_80_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: %R crosses below -20 AND 1w trending (ADX > 25) AND volume spike
            elif (williams_cross_20_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: %R crosses -50 (mean reversion) or opposite signal
            exit_long = (position == 1 and 
                        (williams_cross_50_down or williams_cross_20_down))
            exit_short = (position == -1 and 
                         (williams_cross_50_up or williams_cross_80_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals