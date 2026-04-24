#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for regime filter (high ATR = trending market, low ATR = range-bound).
- Entry: Long when close breaks above Donchian upper band AND ATR(1d) > ATR(1d) MA(50) AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below Donchian lower band AND ATR(1d) > ATR(1d) MA(50) AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR trailing stop (highest high since entry - 3.0 * ATR(4h) for long, lowest low since entry + 3.0 * ATR(4h) for short).
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for structure, volume confirmation for participation,
  1d ATR regime filter to avoid ranging markets, and ATR trailing stop for risk management.
- Designed to work in both bull and bear markets via regime filter and tight entry conditions.
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
    
    # Get 4h data for Donchian channels, volume MA(20) and ATR(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) for 4h timeframe
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 4h timeframe (for trailing stop)
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for 1d timeframe (regime filter)
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])  # first TR is high-low
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR(1d) MA(50) for regime filter threshold
    atr_ma_50_1d = pd.Series(atr14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR and ATR MA to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Donchian needs 20, ATR needs 14, ATR MA needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr14_4h[i]) or 
            np.isnan(vol_ma_4h[i]) or 
            np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr_ma_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr_4h = atr14_4h[i]
        
        # Regime filter: 1d ATR > 1d ATR MA(50) indicates trending market
        regime_filter = atr14_1d_aligned[i] > atr_ma_50_1d_aligned[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if regime_filter and vol_confirm:
                # Long: Close breaks above Donchian upper band
                if curr_close > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_high
                # Short: Close breaks below Donchian lower band
                elif curr_close < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: highest high since entry - 3.0 * ATR(4h)
            stoploss = highest_since_entry - 3.0 * curr_atr_4h
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: lowest low since entry + 3.0 * ATR(4h)
            stoploss = lowest_since_entry + 3.0 * curr_atr_4h
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeSpike_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0