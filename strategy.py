#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above upper Donchian AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below lower Donchian AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit on opposite Donchian level touch or ATR trailing stop (2.0*ATR).
Uses 1d HTF for trend alignment. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
Donchian channels provide robust structure; breakouts with volume and trend filter capture strong moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels from previous 12h bar (using 12h data)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian levels for each 12h bar (20-period)
    high_ma = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (12h -> 12h, no change but for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_ma)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_ma)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # vol_ma (20), ema_34_1d (34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_1d_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Break above upper Donchian AND price > 1d EMA34 AND volume spike
            if price > upper and close[i] > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below lower Donchian AND price < 1d EMA34 AND volume spike
            elif price < lower and close[i] < ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Touch opposite Donchian level (lower for longs, upper for shorts)
            if position == 1 and price < lower:  # Price breaks below lower Donchian (long exit)
                exit_signal = True
            elif position == -1 and price > upper:  # Price breaks above upper Donchian (short exit)
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation_LevelExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0