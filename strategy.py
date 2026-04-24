#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w Camarilla H3/L3 for trend direction (bullish if price > H3, bearish if price < L3).
- Entry: Long when price breaks above prior 6h Donchian(20) high AND price > 1w H3 AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior 6h Donchian(20) low AND price < 1w L3 AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 6h Donchian(20) low,
        exit short when price crosses above prior 6h Donchian(20) high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Uses weekly Camarilla pivot as structural filter to avoid counter-trend trades in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate prior 1w Camarilla levels (H3, L3) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla calculations for H3/L3 levels
    rang_1w = high_1w - low_1w
    camarilla_h3_1w = close_1w + rang_1w * 1.1 / 4
    camarilla_l3_1w = close_1w - rang_1w * 1.1 / 4
    
    # Align 1w Camarilla levels to 6h
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Calculate 6h Donchian(20) channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above 6h Donchian high AND price > 1w H3 AND volume confirmed
            if curr_high > donchian_high[i] and curr_close > camarilla_h3_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h Donchian low AND price < 1w L3 AND volume confirmed
            elif curr_low < donchian_low[i] and curr_close < camarilla_l3_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 6h Donchian low
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 6h Donchian high
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarillaH3L3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0