#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1w ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w ADX(14) for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when Elder Bull Power > 0 and Bear Power < 0 in trending regime with volume > 1.5 * 6h volume MA(20);
         Short when Elder Bull Power < 0 and Bear Power > 0 in trending regime with volume > 1.5 * 6h volume MA(20).
- Exit: Opposite Elder Ray signal (Bull Power and Bear Power cross zero) or volume drops below 0.5 * MA.
- Signal size: 0.25 discrete to balance capture and fee control.
- Elder Ray measures bull/bear power relative to EMA13; ADX filters for trending markets; volume confirms conviction.
- Works in bull (strong bull power) and bear (strong bear power) regimes when ADX confirms trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h volume MA(20) for confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 13, 20, 1)  # ADX needs 30, EMA needs 13, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging (we only trade in trending)
        trending_regime = adx_aligned[i] > 25
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = volume[i] > 1.5 * vol_ma_6h[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0 and bear_power[i] < 0  # Bullish: strong bull power, weak bear power
        bear_signal = bull_power[i] < 0 and bear_power[i] > 0  # Bearish: weak bull power, strong bear power
        exit_signal = (bull_power[i] * bear_power[i] >= 0) or volume[i] < 0.5 * vol_ma_6h[i]  # Both same sign or low volume
        
        if position == 0:
            # Check for entry signals
            if bull_signal and trending_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            elif bear_signal and trending_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: check exit conditions
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0