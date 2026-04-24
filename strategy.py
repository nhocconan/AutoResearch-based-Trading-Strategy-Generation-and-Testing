#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX Momentum with 1d ADX Regime Filter and Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when TRIX(12) crosses above zero AND ADX > 25 AND volume > 2.0 * 4h volume MA(20);
         Short when TRIX(12) crosses below zero AND ADX > 25 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when TRIX crosses below zero; Short exits when TRIX crosses above zero.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (trend continuation) and bear (trend continuation) with regime filter preventing whipsaws in ranging markets.
- Uses TRIX for smooth momentum and ADX to avoid low-momentum environments.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR and DM
    tr_sum = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_sum = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_di_sum = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_sum / tr_sum
    minus_di = 100 * minus_di_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    
    # ADX
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate TRIX(12) on 4h
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 12, 20)  # ADX needs 30, TRIX needs 12, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_trix = trix[i]
        prev_trix = trix[i-1]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        # Regime filter: ADX > 25 for trending market
        regime_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and regime_filter:
                # Long: TRIX crosses above zero (bullish momentum)
                if prev_trix <= 0 and curr_trix > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero (bearish momentum)
                elif prev_trix >= 0 and curr_trix < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when TRIX crosses below zero
            if curr_trix < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TRIX crosses above zero
            if curr_trix > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Momentum_1dADX_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0