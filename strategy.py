#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d/1w data for weekly Camarilla pivot direction (price > weekly H3 = bullish bias, price < weekly L3 = bearish bias).
- Entry: Long when price breaks above 6h Donchian upper(20) AND price > weekly H3 AND volume > 1.8 * 6h volume MA(20);
         Short when price breaks below 6h Donchian lower(20) AND price < weekly L3 AND volume > 1.8 * 6h volume MA(20).
- Exit: Close below/above Donchian lower/upper(20) for profit-taking, with ATR-based stoploss (2.5 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channel for structure, weekly pivot bias for trend filter, volume confirmation for participation,
  and ATR for risk management. Designed to work in both bull and bear markets via weekly pivot filter.
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
    
    # Get 6h data for Donchian(20), volume MA(20), and ATR(14)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR(14) for 6h timeframe
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_6h[0] - low_6h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) for 6h timeframe
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for Camarilla pivot levels (H3, L3) as trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # need at least a few weeks
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels from prior 1w OHLC
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    camarilla_h3_1w = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_l3_1w = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14)  # Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(camarilla_h3_1w_aligned[i]) or 
            np.isnan(camarilla_l3_1w_aligned[i]) or 
            np.isnan(vol_ma_6h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 1.8x threshold for balanced entry frequency
        vol_confirm = curr_volume > 1.8 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Donchian upper(20) AND price > weekly H3 (bullish bias)
                if curr_high > donchian_upper[i] and curr_close > camarilla_h3_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Price breaks below Donchian lower(20) AND price < weekly L3 (bearish bias)
                elif curr_low < donchian_lower[i] and curr_close < camarilla_l3_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr
            # Profit take: close below Donchian lower(20)
            if curr_close < stoploss or curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr
            # Profit take: close above Donchian upper(20)
            if curr_close > stoploss or curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyCamarillaH3L3_Bias_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0