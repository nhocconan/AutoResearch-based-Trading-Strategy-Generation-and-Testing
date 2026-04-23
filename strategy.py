#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above upper BB AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
Short when price breaks below lower BB AND 1d ADX > 25 AND volume > 1.5x 20-period MA.
Exit when price returns to middle BB (20-period SMA).
Uses 1d HTF for ADX trend filter to ensure we only trade in strong trends, avoiding whipsaws in ranging markets.
BB breakouts capture momentum, ADX ensures trend strength, volume confirms participation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    middle_bb = sma_20  # 20-period SMA for exit
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement (+DM and -DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Handle first values
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    adx_period = 14
    alpha = 1.0 / adx_period
    
    atr = np.full_like(tr, np.nan)
    atr[adx_period] = np.nanmean(tr[1:adx_period+1])  # Initial ATR
    
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    adx = np.full_like(tr, np.nan)
    
    # Initial values for smoothing
    if not np.isnan(atr[adx_period]):
        atr_smoothed = atr[adx_period]
        plus_dm_smoothed = np.nansum(plus_dm[1:adx_period+1])
        minus_dm_smoothed = np.nansum(minus_dm[1:adx_period+1])
        
        plus_di[adx_period] = (plus_dm_smoothed / atr_smoothed) * 100 if atr_smoothed != 0 else 0
        minus_di[adx_period] = (minus_dm_smoothed / atr_smoothed) * 100 if atr_smoothed != 0 else 0
        dx[adx_period] = (np.abs(plus_di[adx_period] - minus_di[adx_period]) / 
                         (plus_di[adx_period] + minus_di[adx_period])) * 100 if (plus_di[adx_period] + minus_di[adx_period]) != 0 else 0
        
        # Wilder's smoothing for subsequent values
        for i in range(adx_period + 1, len(tr)):
            atr_smoothed = (atr_smoothed * (adx_period - 1) + tr[i]) / adx_period
            plus_dm_smoothed = (plus_dm_smoothed * (adx_period - 1) + plus_dm[i]) / adx_period
            minus_dm_smoothed = (minus_dm_smoothed * (adx_period - 1) + minus_dm[i]) / adx_period
            
            plus_di_val = (plus_dm_smoothed / atr_smoothed) * 100 if atr_smoothed != 0 else 0
            minus_di_val = (minus_dm_smoothed / atr_smoothed) * 100 if atr_smoothed != 0 else 0
            plus_di[i] = plus_di_val
            minus_di[i] = minus_di_val
            dx_val = (np.abs(plus_di_val - minus_di_val) / (plus_di_val + minus_di_val)) * 100 if (plus_di_val + minus_di_val) != 0 else 0
            dx[i] = dx_val
            
            # ADX is EMA of DX
            if np.isnan(adx[i-1]):
                adx[i] = dx[i]
            else:
                adx[i] = (dx[i] + (adx_period - 1) * adx[i-1]) / adx_period
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, adx_period*2)  # BB, volume MA, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 1.5x 20-period MA (higher threshold to reduce frequency)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX filter: trend strength > 25
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper BB AND ADX filter AND volume filter
            if close[i] > upper_bb[i] and adx_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND ADX filter AND volume filter
            elif close[i] < lower_bb[i] and adx_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price returns to middle BB (20-period SMA)
                if close[i] < middle_bb[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price returns to middle BB (20-period SMA)
                if close[i] > middle_bb[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BollingerBreakout_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0