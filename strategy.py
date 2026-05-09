#!/usr/bin/env python3
# Hypothesis: 6h price action with 12h ADX trend strength and 1d volume confirmation
# Long when price > 12h VWAP, ADX > 25 (trending), and volume > 1.5x 20-period average
# Short when price < 12h VWAP, ADX > 25, and volume > 1.5x 20-period average
# Exit when ADX < 20 (trend weakening) or price crosses VWAP
# Uses ADX for trend filter, VWAP for dynamic support/resistance, volume for conviction
# Designed to capture strong trends while avoiding choppy markets
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "6h_ADX_VWAP_Volume_Trend"
timeframe = "6h"
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
    
    # Calculate VWAP for 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Typical price for VWAP
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # VWAP calculation: cumulative(TP * Volume) / cumulative(Volume)
    vwap_12h = (typical_price * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h = vwap_12h.values
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate ADX for 12h timeframe (14-period)
    if len(df_12h) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_12h['high'] - df_12h['high'].shift(1)
    down_move = df_12h['low'].shift(1) - df_12h['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_12h = 100 * plus_dm_14 / tr_14
    minus_di_12h = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above VWAP, strong trend (ADX > 25), volume confirmation
            if (close[i] > vwap_12h_aligned[i] and 
                adx_12h_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP, strong trend (ADX > 25), volume confirmation
            elif (close[i] < vwap_12h_aligned[i] and 
                  adx_12h_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakening (ADX < 20) or price crosses below VWAP
            if (adx_12h_aligned[i] < 20) or (close[i] < vwap_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakening (ADX < 20) or price crosses above VWAP
            if (adx_12h_aligned[i] < 20) or (close[i] > vwap_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals