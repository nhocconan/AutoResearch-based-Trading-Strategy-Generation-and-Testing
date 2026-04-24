#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Donchian(20) upper band AND price > 12h EMA50 AND volume > 1.5 * 4h volume MA(20);
         Short when close breaks below Donchian(20) lower band AND price < 12h EMA50 AND volume > 1.5 * 4h volume MA(20).
- Exit: ATR trailing stop (highest high since entry - 3.0 * ATR(14) for long, lowest low since entry + 3.0 * ATR(14) for short).
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channels for structure, volume confirmation for participation,
  12h EMA50 trend filter to avoid counter-trend trades, and ATR trailing stop for risk management.
- Designed to work in both bull and bear markets via trend filter and tight entry conditions.
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
    
    # Get 4h data for Donchian(20) and volume MA(20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian(20) bands for 4h timeframe
    donchian_20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h Donchian bands to 15m timeframe (if needed) - but we're using 4h as primary
    # Since timeframe is 4h, we don't need alignment for 4h indicators
    donchian_20_high_aligned = donchian_20_high
    donchian_20_low_aligned = donchian_20_low
    vol_ma_4h_aligned = vol_ma_4h
    atr14_aligned = atr14
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
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
            np.isnan(donchian_20_high_aligned[i]) or 
            np.isnan(donchian_20_low_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(atr14_aligned[i])):
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
        curr_atr = atr14_aligned[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above Donchian(20) upper band AND price > 12h EMA50 (uptrend)
                if curr_close > donchian_20_high_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_high_since_entry = curr_high
                # Short: Close breaks below Donchian(20) lower band AND price < 12h EMA50 (downtrend)
                elif curr_close < donchian_20_low_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_low_since_entry = curr_low
        elif position == 1:
            # Long position: update highest high and check ATR trailing stop
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # ATR trailing stop: highest high since entry - 3.0 * ATR
            stoploss = highest_high_since_entry - 3.0 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check ATR trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # ATR trailing stop: lowest low since entry + 3.0 * ATR
            stoploss = lowest_low_since_entry + 3.0 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0