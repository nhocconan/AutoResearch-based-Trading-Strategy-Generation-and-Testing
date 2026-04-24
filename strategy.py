#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian(20) upper band AND price > 1w EMA50 AND volume > 2.0 * 12h volume MA(30);
         Short when price breaks below Donchian(20) lower band AND price < 1w EMA50 AND volume > 2.0 * 12h volume MA(30).
- Exit: ATR-based stoploss (2.0 * ATR(14)) and time-based exit (hold max 6 bars = 3 days).
- Signal size: 0.25 discrete to control fee drag.
- Designed to capture strong trends with volume confirmation while avoiding choppy markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian, volume MA, and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) bands for 12h timeframe
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 12h timeframe
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(30) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 20, 14)  # EMA50 needs 50, Donchian needs 20, volume MA needs 30, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian upper band AND price > 1w EMA50 (uptrend)
                if curr_high > upper_20[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    bars_since_entry = 0
                # Short: price breaks below Donchian lower band AND price < 1w EMA50 (downtrend)
                elif curr_low < lower_20[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    bars_since_entry = 0
        elif position != 0:
            # Update bars since entry
            bars_since_entry += 1
            
            # Check exit conditions
            exit_signal = False
            
            if position == 1:
                # Long position: check exit conditions
                # Stoploss: 2.0 * ATR below entry
                stoploss = entry_price - 2.0 * curr_atr
                # Time-based exit: hold max 6 bars (3 days)
                if curr_low < stoploss or bars_since_entry >= 6:
                    exit_signal = True
            else:  # position == -1
                # Short position: check exit conditions
                # Stoploss: 2.0 * ATR above entry
                stoploss = entry_price + 2.0 * curr_atr
                # Time-based exit: hold max 6 bars (3 days)
                if curr_high > stoploss or bars_since_entry >= 6:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0