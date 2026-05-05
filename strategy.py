#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime + Volume Spike
# Elder Ray measures bull/bear power (close - EMA13 for bull, EMA13 - close for bear)
# ADX > 25 indicates trending market, < 20 indicates ranging
# In trending markets (ADX > 25): go long when bull power > 0 and rising, short when bear power > 0 and rising
# In ranging markets (ADX < 20): fade extremes - short when bull power > 0 and falling, long when bear power > 0 and rising
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Works in bull (trend following) and bear (mean reversion in ranges) markets
# Timeframe: 6h (primary timeframe as required)

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA13 and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray components
    bull_power = close - ema_13_1d_aligned  # close - EMA13
    bear_power = ema_13_1d_aligned - close  # EMA13 - close
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry
            if adx_aligned[i] > 25:  # Trending market
                # Long: bull power positive and rising (momentum building)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bear power positive and rising (momentum building)
                elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:  # Ranging market
                # Long: bear power positive but fading (selling exhaustion)
                if bear_power[i] > 0 and bear_power[i] < bear_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bull power positive but fading (buying exhaustion)
                elif bull_power[i] > 0 and bull_power[i] < bull_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend weakening or regime change
            if adx_aligned[i] < 20 or bull_power[i] < 0 or (bull_power[i] > 0 and bull_power[i] < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakening or regime change
            if adx_aligned[i] < 20 or bear_power[i] < 0 or (bear_power[i] > 0 and bear_power[i] < bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals