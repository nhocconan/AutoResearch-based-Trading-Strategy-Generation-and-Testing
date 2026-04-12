#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_breakout_v1
# Daily timeframe strategy using weekly Camarilla pivot levels with volume confirmation and ADX trend filter.
# Captures institutional breakouts in trending markets while avoiding false signals in ranging conditions.
# Designed to work in both bull and bear markets by using symmetric long/short logic.
# Target: 15-25 trades/year per symbol for minimal fee drag.

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to daily timeframe
    h3_level = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    h4_level = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: volume > 1.8 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # ADX trend filter: only trade when ADX > 25 (trending market)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smooth TR and DM
    tr_period = 14
    atr = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Pad arrays to match length
    adx_padded = np.concatenate([np.full(tr_period, np.nan), adx])
    adx_filter = adx_padded > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and ADX filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H3 with volume and trend
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3 with volume and trend
        elif close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite H4/L4 breakout
        elif close[i] > h4_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        elif close[i] < l4_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals