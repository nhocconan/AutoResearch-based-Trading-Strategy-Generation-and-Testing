#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop
Hypothesis: 4h Donchian(20) breakouts with 12h EMA50 trend filter and volume confirmation. Uses ATR-based trailing stop (2.5x ATR) for risk management. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year). Works in both bull and bear markets by following 12h trend direction while using Donchian channels for breakout signals. Volume confirmation reduces false breakouts. Discrete position sizing (0.25) minimizes fee drag.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian(20) channels
    donchian_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need Donchian(20), EMA50(12h), volume avg(30), ATR(14)
    start_idx = max(donchian_period, 50, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 12h EMA50 trend filter AND volume confirmation
            # Long: price closes above upper band AND above EMA50 (12h uptrend) AND volume confirmation
            long_condition = (close_val > upper_band) and (close_val > ema_val) and vol_conf
            # Short: price closes below lower band AND below EMA50 (12h downtrend) AND volume confirmation
            short_condition = (close_val < lower_band) and (close_val < ema_val) and vol_conf
            
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
            # 1. Price touches lower Donchian band (opposite breakout)
            # 2. 12h EMA50 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry
            exit_condition = (close_val < lower_band) or (close_val < ema_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches upper Donchian band (opposite breakout)
            # 2. 12h EMA50 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry
            exit_condition = (close_val > upper_band) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0