#!/usr/bin/env python3
"""
4h_12h_Market_Regime_Adaptive_Strategy_v1
Hypothesis: In trending markets (ADX > 25), follow 12h EMA trend; in ranging markets (ADX < 20), 
mean-revert at Bollinger Bands (20,2) on 4h. Uses volume confirmation to avoid false signals.
Designed to work in both bull and bear markets by adapting to regime.
Target: 20-40 trades per year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Market_Regime_Adaptive_Strategy_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for Bollinger Bands (more stable than 4h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 12h EMA (21) for trend direction ===
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === 12h ADX (14) for regime detection ===
    # TR calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = WilderSmooth(tr, 14)
    plus_dm_smoothed = WilderSmooth(plus_dm, 14)
    minus_dm_smoothed = WilderSmooth(minus_dm, 14)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / (tr_smoothed + 1e-10)
    minus_di = 100 * minus_dm_smoothed / (tr_smoothed + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = WilderSmooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 4h Bollinger Bands (20,2) for mean reversion ===
    sma_20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_4h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_4h + (2 * std_20_4h)
    lower_bb = sma_20_4h - (2 * std_20_4h)
    
    # === Volume confirmation (1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(sma_20_4h[i]) or np.isnan(std_20_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        ema_trend = ema_21_12h_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        vol_ok = vol_confirm[i]
        
        # Regime-based logic
        if adx_val > 25:  # Trending regime - follow 12h EMA
            # Long when price above EMA, short when below
            long_signal = price > ema_trend and vol_ok
            short_signal = price < ema_trend and vol_ok
            
            # Exit when price crosses back
            if position == 1 and price <= ema_trend:
                position = 0
                signals[i] = 0.0
            elif position == -1 and price >= ema_trend:
                position = 0
                signals[i] = 0.0
            elif long_signal and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_signal and position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
                
        elif adx_val < 20:  # Ranging regime - mean reversion at BB
            # Long at lower BB, short at upper BB
            long_signal = price <= lower and vol_ok
            short_signal = price >= upper and vol_ok
            
            # Exit when price returns to middle (SMA)
            middle = sma_20_4h[i]
            if position == 1 and price >= middle:
                position = 0
                signals[i] = 0.0
            elif position == -1 and price <= middle:
                position = 0
                signals[i] = 0.0
            elif long_signal and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_signal and position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:  # Transition regime (20 <= ADX <= 25) - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals