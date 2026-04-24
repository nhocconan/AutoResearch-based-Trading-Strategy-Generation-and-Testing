#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Camarilla H3/L3 for trend direction (bullish if close > weekly H3, bearish if close < weekly L3).
- Donchian channel: Calculated from prior 6h OHLC (upper/lower bands for breakout).
- Entry: Long when price breaks above prior 6h Donchian upper AND weekly close > weekly H3 AND volume > 1.8 * volume MA(30).
         Short when price breaks below prior 6h Donchian lower AND weekly close < weekly L3 AND volume > 1.8 * volume MA(30).
- Exit: Close-based reversal - exit long when price crosses below prior 6h Donchian lower,
        exit short when price crosses above prior 6h Donchian upper.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Uses weekly Camarilla pivot trend filter (proven edge from DB top performers) for BTC/ETH/SOL.
Works in both bull and bear markets by aligning with higher-timeframe structure.
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
    
    # Get 1w data for weekly Camarilla pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (H3, L3) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    rang_1w = high_1w - low_1w
    camarilla_h3_1w = close_1w + rang_1w * 1.1 / 4
    camarilla_l3_1w = close_1w - rang_1w * 1.1 / 4
    
    # Align weekly Camarilla levels to 6h
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Get 6h data for Donchian channel (need 20 periods for calculation)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channel (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian upper: max(high, 20)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: min(low, 20)
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Calculate volume MA(30) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 30)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Long: Price breaks above prior 6h Donchian upper AND weekly close > weekly H3 AND volume confirmed
            if curr_close > donchian_upper_aligned[i] and curr_close > camarilla_h3_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 6h Donchian lower AND weekly close < weekly L3 AND volume confirmed
            elif curr_close < donchian_lower_aligned[i] and curr_close < camarilla_l3_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 6h Donchian lower (reversion to mean)
            if curr_close < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 6h Donchian upper (reversion to mean)
            if curr_close > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarillaH3L3_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0