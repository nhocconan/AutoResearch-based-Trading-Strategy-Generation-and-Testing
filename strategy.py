#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to target 7-25 trades/year on 1d timeframe.
1d timeframe reduces noise and fee drag while capturing major trends.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels from 1d timeframe (20-period)
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    close_1d = get_htf_data(prices, '1d')['close'].values
    
    # Donchian upper/lower bands (20-period high/low)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 1d timeframe (prices is already 1d)
    upper_aligned = upper_20  # already aligned since we loaded 1d data directly
    lower_aligned = lower_20
    midpoint_aligned = midpoint_20
    
    # Volume average (20-period for 1d timeframe)
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
    start_idx = max(34, 20, 20, 14)  # EMA34 needs 34, Donchian needs 20, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1w_aligned[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        midpoint = midpoint_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend (price > EMA34) AND volume spike (2.0x avg)
            if close[i] > upper and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower band AND downtrend (price < EMA34) AND volume spike (2.0x avg)
            elif close[i] < lower and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= midpoint:
                exit_signal = True
            elif position == -1 and close[i] >= midpoint:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0