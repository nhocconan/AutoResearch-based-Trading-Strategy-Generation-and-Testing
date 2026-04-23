#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price > SMA50) and volume confirmation (>1.5x 20-bar avg volume).
Long when price breaks above Donchian upper channel AND 1d close > 1d SMA50 AND volume spike.
Short when price breaks below Donchian lower channel AND 1d close < 1d SMA50 AND volume spike.
Exit when price crosses Donchian midline (median of upper/lower) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-50 trades/year on 4h timeframe.
Donchian channels provide structural breakouts; 1d SMA50 filters trend direction; volume confirms conviction.
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
    
    # Calculate 1d SMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for SMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
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
    start_idx = max(lookback, 20, 50)  # donchian20, vol_ma20, sma50_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        sma_val = sma_50_1d_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        midline = donchian_mid[i]
        
        if position == 0:
            # Long: Break above upper channel AND bullish trend (1d close > SMA50) AND volume spike
            if price > upper and close[i] > sma_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below lower channel AND bearish trend (1d close < SMA50) AND volume spike
            elif price < lower and close[i] < sma_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Price crosses Donchian midline
            if position == 1 and price < midline:
                exit_signal = True
            elif position == -1 and price > midline:
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

name = "4H_Donchian20_Breakout_1dSMA50_Trend_VolumeSpike_MidlineExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0