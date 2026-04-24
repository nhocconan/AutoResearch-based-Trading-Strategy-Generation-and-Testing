#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian channel breakout with 4h ADX trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h ADX(14) > 25 for trending regime (filters choppy markets).
- Donchian channels from 20-period: Upper = highest high(20), Lower = lowest low(20).
- Entry: Long when close breaks above Upper AND ADX > 25 AND volume > 1.5 * 1h volume MA(20);
         Short when close breaks below Lower AND ADX > 25 AND volume > 1.5 * 1h volume MA(20).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.20 discrete to control fee drag.
- Uses 4h ADX to avoid whipsaws in ranging markets; volume ensures conviction.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
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
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14) for trend filter
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[high_4h[0] - low_4h[0]], tr_4h])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 1h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 1h timeframe
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian needs 20, ATR needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_1h[i]) or 
            not in_session[i]):
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
        curr_atr = atr14[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if adx_aligned[i] > 25 and vol_confirm:
                # Long: Close breaks above upper Donchian AND ADX > 25 (uptrend)
                if curr_close > highest_20[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Close breaks below lower Donchian AND ADX > 25 (downtrend)
                elif curr_close < lowest_20[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.0 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hADX_Trend_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0