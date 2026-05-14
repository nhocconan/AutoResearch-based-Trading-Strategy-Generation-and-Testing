#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) and volume filter to target 20-50 trades/year.
4h timeframe balances noise reduction and trade frequency, proven effective on BTC/ETH in both bull/bear regimes.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels from 4h timeframe (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Donchian upper/lower bands (20-period high/low)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_4h = (upper_4h + lower_4h) / 2.0
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), upper_4h)
    lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), lower_4h)
    midpoint_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), midpoint_4h)
    
    # Volume average (20-period for 4h timeframe)
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
    start_idx = max(34, 20, 20)  # EMA34 needs 34, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
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

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0