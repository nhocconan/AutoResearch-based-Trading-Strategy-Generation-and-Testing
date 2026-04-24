#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for regime filter (ATR ratio > 1.2 = trending market, < 0.8 = ranging market).
- Entry: Long when price breaks above Donchian(20) high AND ATR ratio > 1.2 AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian(20) low AND ATR ratio > 1.2 AND volume > 1.5 * 12h volume MA(20).
- Exit: ATR trailing stop (3.0 * ATR) from highest high/lowest low since entry.
- Signal size: 0.25 discrete to control fee drag.
- Designed to capture strong trending moves while avoiding choppy markets via ATR regime filter.
- Works in both bull and bear markets by taking breakouts in direction of prevailing volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels, volume MA(20), and ATR(14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR(14) for 12h timeframe
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels for 12h timeframe
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR(14) regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for 1d timeframe
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])  # first TR is high-low
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 12h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate ATR ratio (12h ATR / 1d ATR) for regime filter
    atr_ratio = np.where(atr14_1d_aligned > 0, atr14_12h / atr14_1d_aligned, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(atr14_12h[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14_12h[i]
        curr_atr_ratio = atr_ratio[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_12h[i]
        
        # Regime filter: trending market (ATR ratio > 1.2)
        trending_regime = curr_atr_ratio > 1.2
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and trending_regime:
                # Long: price breaks above Donchian(20) high
                if curr_close > highest_20[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high = curr_high
                    lowest_low = curr_low
                # Short: price breaks below Donchian(20) low
                elif curr_close < lowest_20[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    highest_high = curr_high
                    lowest_low = curr_low
        elif position == 1:
            # Long position: update highest high and check ATR trailing stop
            highest_high = max(highest_high, curr_high)
            lowest_low = min(lowest_low, curr_low)
            
            # ATR trailing stop: 3.0 * ATR below highest high
            stoploss = highest_high - 3.0 * curr_atr
            
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check ATR trailing stop
            highest_high = max(highest_high, curr_high)
            lowest_low = min(lowest_low, curr_low)
            
            # ATR trailing stop: 3.0 * ATR above lowest low
            stoploss = lowest_low + 3.0 * curr_atr
            
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high = 0.0
                lowest_low = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Regime_VolumeSpike_ATRTrail_v1"
timeframe = "12h"
leverage = 1.0