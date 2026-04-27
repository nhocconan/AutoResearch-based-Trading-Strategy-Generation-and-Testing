#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATR_Filter
Hypothesis: 4h Donchian(20) breakout in direction of 1d EMA50 trend with volume confirmation and ATR-based volatility filter. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year). Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear markets by following 1d trend direction. Volume spike (>1.5x 20-bar avg) confirms breakout strength. ATR filter avoids low-volatility false breakouts. ATR trailing stop manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Volatility filter: ATR > 0.5 * 50-period ATR average (avoid low-vol chop)
    atr_avg = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > (0.5 * atr_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need Donchian(20), EMA50(50), ATR(14), vol_avg(20), atr_avg(50)
    start_idx = max(20, 50, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_conf = volume_confirm[i]
        vol_ok = vol_filter[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of 1d EMA50 trend with volume and volatility filters
            # Long: price closes above upper channel AND above EMA50 (1d uptrend) AND volume spike AND adequate volatility
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf and vol_ok
            # Short: price closes below lower channel AND below EMA50 (1d downtrend) AND volume spike AND adequate volatility
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf and vol_ok
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches lower Donchian channel (opposite breakout)
            # 2. 1d EMA50 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry
            exit_condition = (close_val < lower_channel) or (close_val < ema_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches upper Donchian channel (opposite breakout)
            # 2. 1d EMA50 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry
            exit_condition = (close_val > upper_channel) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATR_Filter"
timeframe = "4h"
leverage = 1.0