#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla H3 level (1d) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level (1d) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses Camarilla Pivot point (PP) or opposite signal occurs
# - Camarilla levels provide intraday support/resistance based on previous day's range
# - 1d ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 19-50 trades/year on 4h (75-200 total over 4 years) to avoid fee drag

name = "4h_1d_camarilla_breakout_volume_adx_v1"
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
    
    # Pre-compute 4h Camarilla levels (based on previous 1d candle)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # We'll calculate Camarilla levels for each 4h bar using the previous completed 1d bar
    # To do this, we need to get the previous 1d bar's OHLC for each 4h bar
    camarilla_pp = np.full_like(close_4h, np.nan, dtype=float)
    camarilla_h3 = np.full_like(close_4h, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close_4h, np.nan, dtype=float)
    
    # For each 4h bar, find the previous completed 1d bar
    for i in range(len(prices)):
        if i < 6:  # Need at least 6 bars to have previous 1d (4h bars per day = 6)
            continue
            
        # Get current time
        current_time = prices['open_time'].iloc[i]
        
        # Find previous completed 1d bar (strictly before current_time)
        # Since df_1d is indexed by open_time, we can find the last 1d bar that closed before current_time
        prev_1d_mask = df_1d['open_time'] < current_time
        if not prev_1d_mask.any():
            continue
            
        prev_1d_idx = prev_1d_mask.idxmax()  # Get the index of the last True
        prev_1d = df_1d.loc[prev_1d_idx]
        
        high_prev = prev_1d['high']
        low_prev = prev_1d['low']
        close_prev = prev_1d['close']
        
        # Calculate Camarilla levels
        range_prev = high_prev - low_prev
        camarilla_pp[i] = (high_prev + low_prev + close_prev) / 3
        camarilla_h3[i] = camarilla_pp[i] + (range_prev * 1.1 / 4)
        camarilla_l3[i] = camarilla_pp[i] - (range_prev * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_4h = prices['volume'].values
        vol_ma_20 = np.full_like(vol_4h, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_4h[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_4h[i] > 1.5 * vol_ma_20[i]
        
        close_now = close_4h[i]
        camarilla_pp_now = camarilla_pp[i]
        camarilla_h3_now = camarilla_h3[i]
        camarilla_l3_now = camarilla_l3[i]
        adx_now = adx_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close_now > camarilla_h3_now  # price breaks above H3
        breakout_down = close_now < camarilla_l3_now  # price breaks below L3
        pivot_cross_up = (close_4h[i-1] <= camarilla_pp_now and close_now > camarilla_pp_now)  # crosses above PP
        pivot_cross_down = (close_4h[i-1] >= camarilla_pp_now and close_now < camarilla_pp_now)  # crosses below PP
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 1d trending (ADX > 25) AND volume spike
            if (breakout_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND 1d trending (ADX > 25) AND volume spike
            elif (breakout_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Pivot point (PP) or opposite Camarilla breakout
            exit_long = (position == 1 and 
                        (pivot_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (pivot_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals