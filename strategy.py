#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5 * 4h volume MA(20);
         Short when close breaks below Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5 * 4h volume MA(20).
- Exit: ATR-based stoploss (2.0 * ATR(14)) and trailing stop (highest high since entry - 2.5 * ATR).
- Signal size: 0.25 discrete to control fee drag.
- Designed to work in both bull and bear markets via trend filter and volatility-based exits.
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
    
    # Get 4h data for volume MA(20) and ATR(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) for 4h timeframe
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_4h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry frequency
        vol_confirm = curr_volume > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Donchian upper AND price > 1d EMA50 (uptrend)
                if curr_close > donchian_upper[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
                    lowest_low_since_entry = curr_low
                # Short: Close breaks below Donchian lower AND price < 1d EMA50 (downtrend)
                elif curr_close < donchian_lower[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
                    lowest_low_since_entry = curr_low
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Trailing stop: highest high since entry minus 2.5 * ATR
            trailing_stop = highest_high_since_entry - 2.5 * curr_atr
            # Effective stop is the higher of the two (more protective)
            effective_stop = max(stoploss, trailing_stop)
            
            if curr_close < effective_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Trailing stop: lowest low since entry plus 2.5 * ATR
            trailing_stop = lowest_low_since_entry + 2.5 * curr_atr
            # Effective stop is the lower of the two (more protective)
            effective_stop = min(stoploss, trailing_stop)
            
            if curr_close > effective_stop:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0