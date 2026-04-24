#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for trend filter (ATR rising = trending market, ATR falling = ranging).
- Entry: Long when close breaks above Donchian H20 AND 1d ATR(14) > its 10-bar MA AND volume > 2.0 * 6h volume MA(20);
         Short when close breaks below Donchian L20 AND 1d ATR(14) > its 10-bar MA AND volume > 2.0 * 6h volume MA(20).
- Exit: Close below/above Donchian L20/H20 for profit-taking, with ATR-based stoploss (2.5 * ATR(14) 6h).
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for structure, volume confirmation for participation,
  1d ATR trend filter to ensure trades only in trending markets (works in both bull/bear trends),
  and ATR for risk management.
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
    
    # Get 6h data for Donchian(20), volume MA(20) and ATR(14)
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
    atr14_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) for 6h timeframe
    donchian_h20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_l20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for ATR(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])  # first TR is high-low
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) MA(10) for trend filter (rising ATR = trending market)
    atr_ma_10_1d = pd.Series(atr14_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align 1d ATR and its MA to 6h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 10)  # Donchian needs 20, ATR needs 14, MA needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h20[i]) or 
            np.isnan(donchian_l20[i]) or 
            np.isnan(vol_ma_6h[i]) or 
            np.isnan(atr14_6h[i]) or 
            np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr_ma_10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr_6h = atr14_6h[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        # Trend filter: 1d ATR > its 10-bar MA (indicates trending market)
        trend_filter = atr14_1d_aligned[i] > atr_ma_10_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and trend_filter:
                # Long: Close breaks above Donchian H20
                if curr_close > donchian_h20[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Donchian L20
                elif curr_close < donchian_l20[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr_6h
            # Profit take: close below Donchian L20
            if curr_close < stoploss or curr_close < donchian_l20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr_6h
            # Profit take: close above Donchian H20
            if curr_close > stoploss or curr_close > donchian_h20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0