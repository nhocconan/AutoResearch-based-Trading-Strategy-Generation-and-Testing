#!/usr/bin/env python3
"""
1d_Weekly_Range_Reversal_v1
Hypothesis: Uses weekly RSI extremes with weekly pivot levels and daily volume confirmation to capture mean reversions in ranging markets during bear/bull phases. Works by identifying overbought/oversold conditions at weekly support/resistance levels with volume divergence, filtered by weekly ADX trend strength. Target: 15-25 trades/year to minimize fee decay while capturing high-probability reversals.
"""

name = "1d_Weekly_Range_Reversal_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for RSI, pivot points, and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI (14-period) ---
    rsi_period = 14
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # --- Weekly Pivot Points (Standard) ---
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    r1 = 2 * pivot - df_1w['low']
    s1 = 2 * pivot - df_1w['high']
    r2 = pivot + (df_1w['high'] - df_1w['low'])
    s2 = pivot - (df_1w['high'] - df_1w['low'])
    
    # --- Weekly ADX (14-period) for trend filter ---
    # True Range
    tr1 = pd.Series(df_1w['high']).subtract(df_1w['low']).abs()
    tr2 = pd.Series(df_1w['high']).subtract(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).subtract(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(df_1w['high']).diff()
    dm_minus = pd.Series(df_1w['low']).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Wilder's smoothing
    atr_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1w = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1w_values = adx_1w.values
    
    # Align weekly indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_values)
    
    # --- Daily Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Moderate volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        rsi_val = rsi_aligned[i]
        adx_val = adx_1w_aligned[i]
        
        # Define ranging market (ADX < 25) and trending market (ADX >= 25)
        is_ranging = adx_val < 25
        is_trending = adx_val >= 25
        
        # Weekly extreme conditions
        overbought = rsi_val > 70
        oversold = rsi_val < 30
        
        # Price near weekly support/resistance levels (within 0.5%)
        near_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005
        near_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005
        near_r2 = abs(close[i] - r2_aligned[i]) / r2_aligned[i] < 0.005
        near_s2 = abs(close[i] - s2_aligned[i]) / s2_aligned[i] < 0.005
        
        # Mean reversion signals in ranging markets
        if position == 0 and is_ranging:
            # Long setup: oversold + near support + volume spike
            if oversold and (near_s1 or near_s2) and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: overbought + near resistance + volume spike
            elif overbought and (near_r1 or near_r2) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: RSI returns to neutral or hits resistance
            if rsi_val > 50 or near_r1 or near_r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral or hits support
            if rsi_val < 50 or near_s1 or near_s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals