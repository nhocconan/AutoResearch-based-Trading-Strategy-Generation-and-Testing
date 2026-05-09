#!/usr/bin/env python3
# Hypothesis: 12h price action relative to 1d VWAP with 1w ADX trend strength and 1d volume confirmation
# Long when price > 1d VWAP, ADX > 25 (trending), and volume > 1.5x 20-period average
# Short when price < 1d VWAP, ADX > 25, and volume > 1.5x 20-period average
# Exit when ADX < 20 (trend weakening) or price crosses VWAP
# Uses ADX for trend filter, VWAP for dynamic support/resistance, volume for conviction
# Designed to capture strong trends while avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_1dVWAP_1wADX_1dVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Typical price for VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # VWAP calculation: cumulative(TP * Volume) / cumulative(Volume)
    vwap_1d = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate ADX for 1w timeframe (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1w['high'] - df_1w['high'].shift(1)
    down_move = df_1w['low'].shift(1) - df_1w['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_1w = 100 * plus_dm_14 / tr_14
    minus_di_1w = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above VWAP, strong trend (ADX > 25), volume confirmation
            if (close[i] > vwap_1d_aligned[i] and 
                adx_1w_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP, strong trend (ADX > 25), volume confirmation
            elif (close[i] < vwap_1d_aligned[i] and 
                  adx_1w_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) or price crosses below VWAP
            if (adx_1w_aligned[i] < 20) or (close[i] < vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) or price crosses above VWAP
            if (adx_1w_aligned[i] < 20) or (close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals