#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX Regime Filter + Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume > 1.5 * 6h volume MA(20);
         Short when Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND volume > 1.5 * 6h volume MA(20).
         (Note: Bull Power = high - EMA13, Bear Power = low - EMA13)
- Exit: Long exits when Bull Power <= 0; Short exits when Bear Power >= 0.
- Signal size: 0.25 discrete to balance capture and fee control.
- Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets only; volume confirms conviction.
- Works in bull (buying strength in uptrend) and bear (selling weakness in downtrend) with reduced whipsaws.
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_smooth = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray on 6h (EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = high - EMA13
    bear_power = low - ema13   # Bear Power = low - EMA13
    
    # Get 6h data for volume MA(20)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14, 14)  # EMA13 needs 13, ADX needs 14+14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        prev_bull_power = bull_power[i-1] if i > 0 else 0
        prev_bear_power = bear_power[i-1] if i > 0 else 0
        
        # Regime filter from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Not used in entry but could be for exit
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if trending and vol_confirm:
                # Long: Bull Power > 0 AND Bear Power < 0 (bulls in control)
                if curr_bull_power > 0 and curr_bear_power < 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND Bull Power > 0 (bears in control) - same condition!
                # Actually: Short when Bear Power < 0 AND Bull Power < 0? Let's correct:
                # Bear Power < 0 means bears weak, Bull Power < 0 means bulls weak
                # We want: Short when Bear Power < 0 (bears strong) AND Bull Power < 0 (bulls weak)
                elif curr_bear_power < 0 and curr_bull_power < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Bull Power <= 0 (bulls losing strength)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power >= 0 (bears losing strength)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0