#!/usr/bin/env python3
"""
12h_Daily_Camarilla_Pivot_Bounce_With_Volume_Filter
Hypothesis: In ranging markets (common in 2025), price tends to revert to Camarilla pivot levels (S1/S3/R1/R3).
Long when price bounces from S1/S3 with volume confirmation; short when rejected at R1/R3.
Uses daily ADX to filter out strong trends where mean reversion fails.
Designed for 12h timeframe to target ~25 trades/year with high-conviction mean reversion entries.
Works in sideways markets by capturing reversals at key levels; avoids losses in trends via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    if len(tr) >= period:
        tr_period[period-1] = np.sum(tr[:period])
        dm_plus_period[period-1] = np.sum(dm_plus[:period])
        dm_minus_period[period-1] = np.sum(dm_minus[:period])
        
        for i in range(period, len(tr)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
            dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
            dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Directional Indicators
    di_plus = np.zeros_like(tr)
    di_minus = np.zeros_like(tr)
    dx = np.zeros_like(tr)
    
    for i in range(period-1, len(tr)):
        if tr_period[i] != 0:
            di_plus[i] = 100 * dm_plus_period[i] / tr_period[i]
            di_minus[i] = 100 * dm_minus_period[i] / tr_period[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.zeros_like(tr)
    if len(dx) >= 2*period-1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    c = close
    
    # Resistance levels
    r4 = c + range_val * 1.5000
    r3 = c + range_val * 1.2500
    r2 = c + range_val * 1.1666
    r1 = c + range_val * 1.0833
    
    # Support levels
    s1 = c - range_val * 1.0833
    s2 = c - range_val * 1.1666
    s3 = c - range_val * 1.2500
    s4 = c - range_val * 1.5000
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla levels, ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ADX for trend filter (avoid strong trends)
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Daily Camarilla levels
    r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align Camarilla levels (no extra delay needed - levels are based on closed daily bar)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # ADX filter: only trade when ADX < 25 (ranging market)
        ranging_market = adx_1d_aligned[i] < 25
        
        if position == 0 and ranging_market:
            # Long: price bounces from S1 or S3 with volume confirmation
            long_signal = False
            if price <= s1_1d_aligned[i] * 1.002:  # Allow 0.2% slippage
                long_signal = True
            elif price <= s3_1d_aligned[i] * 1.002:
                long_signal = True
            
            if long_signal and volume_ok:
                signals[i] = 0.25
                position = 1
            
            # Short: price rejected at R1 or R3 with volume confirmation
            elif price >= r1_1d_aligned[i] * 0.998:  # Allow 0.2% slippage
                short_signal = True
            elif price >= r3_1d_aligned[i] * 0.998:
                short_signal = True
            
            if short_signal and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches opposite level or ADX increases (trend emerging)
            if price >= r1_1d_aligned[i] * 0.998 or adx_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches opposite level or ADX increases (trend emerging)
            if price <= s1_1d_aligned[i] * 1.002 or adx_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Daily_Camarilla_Pivot_Bounce_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0