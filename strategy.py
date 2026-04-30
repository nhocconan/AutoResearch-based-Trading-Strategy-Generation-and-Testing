#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) zero-line cross with volume confirmation and 1d ADX regime filter
# TRIX(12) filters noise and identifies momentum shifts with less whipsaw than MACD
# Volume confirmation (>1.4x average) ensures breakout legitimacy with controlled frequency
# 1d ADX > 25 ensures we only trade in trending markets, avoiding choppy conditions
# Works in bull/bear: TRIX catches momentum shifts, volume confirms, ADX avoids false signals in ranges
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_TRIX_ZeroCross_Volume_ADX_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) - Triple Exponential Average
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Percentage change
    
    # Need previous TRIX value to detect zero-cross
    trix_prev = np.roll(trix.values, 1)
    trix_prev[0] = np.nan
    
    # Zero-line cross conditions
    trix_cross_up = (trix_prev <= 0) & (trix.values > 0)   # Cross above zero
    trix_cross_down = (trix_prev >= 0) & (trix.values < 0) # Cross below zero
    
    # Volume confirmation: volume > 1.4x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.4 * vol_ma_20)
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX components
    df_1d_copy = df_1d.copy()
    df_1d_copy['high'] = df_1d['high'].values
    df_1d_copy['low'] = df_1d['low'].values
    df_1d_copy['close'] = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_copy['high'] - df_1d_copy['low']
    tr2 = np.abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    tr3 = np.abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1d_copy['high'] - df_1d_copy['high'].shift(1)
    down_move = df_1d_copy['low'].shift(1) - df_1d_copy['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(trix_prev[i]) or 
            np.isnan(trix.values[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_trix = trix.values[i]
        curr_trix_prev = trix_prev[i]
        curr_volume_confirm = volume_confirm[i]
        curr_adx = adx_aligned[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if curr_adx <= 25:
            # In choppy markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade on TRIX zero-cross with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: TRIX crosses above zero
                if curr_trix_prev <= 0 and curr_trix > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: TRIX crosses below zero
                elif curr_trix_prev >= 0 and curr_trix < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: TRIX crosses below zero (momentum reversal)
            if curr_trix_prev > 0 and curr_trix <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero (momentum reversal)
            if curr_trix_prev < 0 and curr_trix >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals