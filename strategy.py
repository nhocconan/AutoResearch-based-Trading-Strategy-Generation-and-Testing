#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1w ADX regime filter and 1d volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1w ADX > 25 (trending) AND 1d volume > 1.5x 20-period MA.
Short when Bear Power < 0 AND Bull Power > 0 AND 1w ADX > 25 AND 1d volume > 1.5x 20-period MA.
Exit when Elder Ray signals reverse or 1w ADX < 20 (range regime).
Uses 1w HTF for regime to avoid whipsaws, 1d volume for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear power via EMA13, ADX filters trending markets, volume avoids low-momentum signals.
Works in bull (strong uptrend signals) and bear (strong downtrend signals) via regime filter.
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
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Calculate 1w ADX for regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # ADX needs ~30 periods
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.abs(high_1w[0] - low_1w[0])], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    dm_plus_smooth[atr_period-1] = np.mean(dm_plus[:atr_period])
    dm_minus_smooth[atr_period-1] = np.mean(dm_minus[:atr_period])
    
    # Wilder's smoothing
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (atr_period - 1) + dm_plus[i]) / atr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (atr_period - 1) + dm_minus[i]) / atr_period
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
    minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = np.zeros_like(dx)
    adx[2*atr_period-1] = np.mean(dx[atr_period:2*atr_period])
    
    for i in range(2*atr_period, len(dx)):
        adx[i] = (adx[i-1] * (atr_period - 1) + dx[i]) / atr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 1d volume MA (20-period) for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 2*14, 20)  # EMA13, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0
        bear_signal = bear_power[i] < 0
        
        # Regime filter: 1w ADX > 25 = trending market
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Exit threshold with hysteresis
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish bias) AND trending AND volume
            if bull_signal and bear_signal and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 (bearish bias) AND trending AND volume
            elif bear_signal and bull_signal and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Elder Ray turns bearish OR ADX < 20 (ranging)
                if not (bull_signal and bear_signal) or ranging:
                    exit_signal = True
            elif position == -1:
                # Short exit: Elder Ray turns bullish OR ADX < 20 (ranging)
                if not (bull_signal and bear_signal) or ranging:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0