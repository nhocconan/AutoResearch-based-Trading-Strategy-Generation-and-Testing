#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from 1d provide key support/resistance. 
# Breakouts above R3 or below S3 with volume > 1.5x average and ADX > 20 (trending) 
# capture momentum. Works in bull markets via long breakouts and bear markets via 
# short breakdowns. Low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h ATR for ADX calculation (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h +DM and -DM for ADX
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, and TR for ADX
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    
    # Camarilla levels: 
    # H4 = close + 1.1 * (high - low)
    # L4 = close - 1.1 * (high - low)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # H2 = close + 1.1 * (high - low) / 4
    # L2 = close - 1.1 * (high - low) / 4
    # H1 = close + 1.1 * (high - low) / 6
    # L1 = close - 1.1 * (high - low) / 6
    
    # We'll use H3/L3 as breakout levels (more reliable than H4/L4)
    high_low_range = df_1d['high'] - df_1d['low']
    camarilla_h3 = df_1d['close'] + 1.1 * high_low_range / 2
    camarilla_l3 = df_1d['close'] - 1.1 * high_low_range / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx[i] > 20
        
        # Breakout signals
        breakout_up = high[i] > camarilla_h3_aligned[i]
        breakdown_down = low[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H3 AND volume confirmation AND trending market
        if breakout_up and vol_confirm and trending and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND volume confirmation AND trending market
        elif breakdown_down and vol_confirm and trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Camarilla level touch (L3 for long, H3 for short)
        elif position == 1 and low[i] <= camarilla_l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] >= camarilla_h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals