#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h ADX regime filter + 12h Supertrend trend + 4h volume spike for entry.
Long when 12h ADX > 25 (trending) AND price > 12h Supertrend AND 4h volume > 2.0x 20-period average.
Short when 12h ADX > 25 AND price < 12h Supertrend AND 4h volume > 2.0x 20-period average.
Exit when ADX < 20 (trend weakening) or opposite Supertrend signal.
Uses discrete position sizing of 0.25 to limit fee drag and manage drawdown.
Target: 75-200 total trades over 4 years (19-50/year) to avoid overtrading.
Combines trend following (Supertrend) with regime filter (ADX) and volume confirmation for robustness.
Works in both bull and bear markets by only trading strong trends with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX and Supertrend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14)
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_12h + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_12h + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # TR for Supertrend
    tr1_st = high_12h - low_12h
    tr2_st = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3_st = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_st = np.maximum(tr1_st, np.maximum(tr2_st, tr3_st))
    atr_st = pd.Series(tr_st).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_12h + low_12h) / 2 + multiplier * atr_st
    basic_lb = (high_12h + low_12h) / 2 - multiplier * atr_st
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Start from atr_period
    for i in range(atr_period, len(close_12h)):
        if i == atr_period:
            supertrend[i] = basic_lb[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == basic_ub[i-1]:
                if close_12h[i] <= basic_ub[i]:
                    supertrend[i] = basic_ub[i]
                    direction[i] = -1
                else:
                    supertrend[i] = basic_lb[i]
                    direction[i] = 1
            else:
                if close_12h[i] >= basic_lb[i]:
                    supertrend[i] = basic_lb[i]
                    direction[i] = 1
                else:
                    supertrend[i] = basic_ub[i]
                    direction[i] = -1
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)  # Note: using 12h alignment for 4h volume MA approximation
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price above Supertrend, strong trend (ADX > 25), volume confirmation
            if (close[i] > supertrend_aligned[i] and 
                adx_12h_aligned[i] > 25 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend, strong trend (ADX > 25), volume confirmation
            elif (close[i] < supertrend_aligned[i] and 
                  adx_12h_aligned[i] > 25 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Supertrend or trend weakens (ADX < 20)
            if (close[i] < supertrend_aligned[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Supertrend or trend weakens (ADX < 20)
            if (close[i] > supertrend_aligned[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hSupertrend_ADX_Volume"
timeframe = "4h"
leverage = 1.0