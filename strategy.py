#!/usr/bin/env python3
"""
6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
Hypothesis: Mean reversion at extreme Camarilla levels (R4/S4) aligned with daily trend captures reversals in both bull and bear markets, while volume confirmation filters false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14287_6h_camarilla1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = calculate_ema(close_1d, 50)
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 50 for EMA)
    start = max(20, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price returns to EMA or opposite Camarilla level
        if position == 1:  # long position
            if close[i] >= ema_1d_aligned[i] or close[i] <= camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] <= ema_1d_aligned[i] or close[i] >= camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at extreme Camarilla levels with trend and volume confirmation
            # Long at S4 when price > EMA (uptrend) with volume confirmation
            # Short at R4 when price < EMA (downtrend) with volume confirmation
            long_setup = (close[i] <= camarilla_s4_aligned[i]) and (close[i] > ema_1d_aligned[i]) and vol_confirm[i]
            short_setup = (close[i] >= camarilla_r4_aligned[i]) and (close[i] < ema_1d_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals