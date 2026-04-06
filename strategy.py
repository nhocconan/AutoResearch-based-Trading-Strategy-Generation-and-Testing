#!/usr/bin/env python3
"""
12h Bollinger Band Reversal with Volume Confirmation and ADX Trend Filter
Hypothesis: In ranging markets (ADX < 25), price tends to revert from Bollinger Band extremes. 
In trending markets (ADX > 25), we avoid reversals to prevent whipsaws.
Volume spike confirms reversal pressure. Works in both bull/bear by adapting to regime.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14332_12h_bb_reversal_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mid = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume filter: spike above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)  # 150% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = bb_period + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_aligned[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to middle band OR stoploss
            if close[i] >= bb_mid[i] or close[i] <= entry_price - 2.5 * bb_std[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to middle band OR stoploss
            if close[i] <= bb_mid[i] or close[i] >= entry_price + 2.5 * bb_std[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversals: price at BB extremes + volume spike + ranging market (ADX < 25)
            at_bb_upper = close[i] >= bb_upper[i]
            at_bb_lower = close[i] <= bb_lower[i]
            ranging = adx_aligned[i] < 25  # ADX < 25 indicates ranging market
            
            long_setup = at_bb_lower and vol_spike[i] and ranging
            short_setup = at_bb_upper and vol_spike[i] and ranging
            
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