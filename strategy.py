#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# - Williams %R(14) on 6h: oversold < -80, overbought > -20
# - Long when %R crosses above -80 from below AND 1d ADX > 20 (trending or strong range) AND volume > 1.3x 20-period average
# - Short when %R crosses below -20 from above AND 1d ADX > 20 AND volume > 1.3x 20-period average
# - Exit when %R crosses opposite threshold (-20 for long exit, -80 for short exit)
# - Williams %R is a momentum oscillator that identifies overbought/oversold levels
# - 1d ADX filter ensures we only trade when higher timeframe has sufficient momentum
# - Volume confirmation prevents false signals in low participation
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years)

name = "6h_1d_williamsr_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    williams_r = np.full_like(close_6h, np.nan, dtype=float)
    period = 14
    
    for i in range(period - 1, len(high_6h)):
        highest_high = np.max(high_6h[i - period + 1:i + 1])
        lowest_low = np.min(low_6h[i - period + 1:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_6h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
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
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_williams_r = np.full(n, np.nan)  # for crossover detection
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Store previous Williams %R for crossover detection
        if i > 0:
            prev_williams_r[i] = williams_r[i-1]
        else:
            prev_williams_r[i] = np.nan
        
        # Skip if any required data is invalid
        if np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        vol_series = prices['volume'].values
        if i >= 19:
            vol_ma = np.mean(vol_series[i-19:i+1])
            vol_condition = vol_series[i] > 1.3 * vol_ma
        else:
            vol_condition = False  # not enough data for MA
        
        williams_now = williams_r[i]
        williams_prev = prev_williams_r[i]
        
        # Williams %R signals
        williams_bullish_cross = (williams_prev <= -80) and (williams_now > -80)
        williams_bearish_cross = (williams_prev >= -20) and (williams_now < -20)
        williams_long_exit = (williams_prev >= -20) and (williams_now < -20)  # cross below -20
        williams_short_exit = (williams_prev <= -80) and (williams_now > -80)  # cross above -80
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: %R crosses above -80 AND 1d trending (ADX > 20) AND volume spike
            if (williams_bullish_cross and 
                adx_1d_aligned[i] > 20 and vol_condition):
                position = 1
                signals[i] = 0.25
            # Short conditions: %R crosses below -20 AND 1d trending (ADX > 20) AND volume spike
            elif (williams_bearish_cross and 
                  adx_1d_aligned[i] > 20 and vol_condition):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: %R crosses opposite threshold
            exit_long = (position == 1 and williams_long_exit)
            exit_short = (position == -1 and williams_short_exit)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals