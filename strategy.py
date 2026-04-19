#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + weekly ADX trend filter + volume confirmation.
# Donchian breakouts capture momentum in trending markets; ADX > 25 filters for strong trends.
# Volume spike confirms breakout legitimacy. Designed for 12h to reduce trade frequency.
# Entry: Long when close > upper band and ADX > 25 and volume spike; Short when close < lower band and ADX > 25 and volume spike.
# Exit: Opposite Donchian band touch or ADX < 20 (trend weakening).
# Target: 15-30 trades/year to avoid fee drag.

name = "12h_Donchian20_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly ADX for trend strength (uses 14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
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
    
    # Smoothed TR, DM+ , DM-
    tr_period = 14
    tr_sum = np.nansum(tr[-tr_period:]) if len(tr) >= tr_period else np.nan
    dm_plus_sum = np.nansum(dm_plus[-tr_period:]) if len(dm_plus) >= tr_period else np.nan
    dm_minus_sum = np.nansum(dm_minus[-tr_period:]) if len(dm_minus) >= tr_period else np.nan
    
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    if len(tr) >= tr_period:
        atr[tr_period-1] = tr_sum
        dm_plus_smooth[tr_period-1] = dm_plus_sum
        dm_minus_smooth[tr_period-1] = dm_minus_sum
        for i in range(tr_period, len(tr)):
            atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period-1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period-1) + dm_minus[i]) / tr_period
    
    # DI+ and DI-
    di_plus = np.full(len(tr), np.nan)
    di_minus = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    
    for i in range(tr_period-1, len(tr)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX: smoothed DX
    adx = np.full(len(tr), np.nan)
    if len(tr) >= 2*tr_period-1:
        dx_valid = dx[tr_period-1:]
        adx[2*tr_period-2] = np.nanmean(dx_valid[:tr_period])
        for i in range(2*tr_period-1, len(tr)):
            adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (waits for prior day close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with strong trend and volume
            if (close[i] > donchian_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with strong trend and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower band or trend weakens
            if (close[i] < donchian_low_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper band or trend weakens
            if (close[i] > donchian_high_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals