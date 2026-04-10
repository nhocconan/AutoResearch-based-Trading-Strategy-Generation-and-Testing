#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above S4 (Camarilla support level 4) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Short when price breaks below R4 (Camarilla resistance level 4) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses the 1d VWAP or opposite signal occurs
# - Camarilla levels from 1d provide precise intraday support/resistance for 6h entries
# - 1d ADX filter ensures we only trade when higher timeframe is strongly trending
# - Volume confirmation prevents false signals in low participation
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_camarilla_breakout_volume_adx_v1"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day's high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = Close + 1.1 * (High - Low) / 2
    # R3 = Close + 1.1 * (High - Low) / 4
    # S3 = Close - 1.1 * (High - Low) / 4
    # S4 = Close - 1.1 * (High - Low) / 2
    camarilla_r4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_r3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_s3 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_s4 = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + 1.1 * range_val / 2
        camarilla_r3[i] = prev_close + 1.1 * range_val / 4
        camarilla_s3[i] = prev_close - 1.1 * range_val / 4
        camarilla_s4[i] = prev_close - 1.1 * range_val / 2
    
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
    
    # Pre-compute 1d VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vol_1d = df_1d['volume'].values
    pv = typical_price_1d * vol_1d
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(vol_1d)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_6h = prices['volume'].values
        vol_ma_20 = np.full_like(vol_6h, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_20[j] = np.mean(vol_6h[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_20[i]) and vol_6h[i] > 1.5 * vol_ma_20[i]
        
        close_now = prices['close'].values[i]
        high_now = prices['high'].values[i]
        low_now = prices['low'].values[i]
        camarilla_s4 = camarilla_s4_aligned[i]
        camarilla_r4 = camarilla_r4_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        adx_now = adx_1d_aligned[i]
        vwap_now = vwap_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close_now > camarilla_r4  # price breaks above R4
        breakout_down = close_now < camarilla_s4  # price breaks below S4
        vwap_cross_up = (prices['close'].values[i-1] <= vwap_now and close_now > vwap_now)  # crosses above VWAP
        vwap_cross_down = (prices['close'].values[i-1] >= vwap_now and close_now < vwap_now)  # crosses below VWAP
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above R4 AND 1d trending (ADX > 25) AND volume spike
            if (breakout_up and adx_now > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below S4 AND 1d trending (ADX > 25) AND volume spike
            elif (breakout_down and adx_now > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses VWAP (mean reversion) or opposite Camarilla breakout
            exit_long = (position == 1 and 
                        (vwap_cross_down or breakout_down))
            exit_short = (position == -1 and 
                         (vwap_cross_up or breakout_up))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals