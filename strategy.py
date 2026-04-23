#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band and close > 12h EMA50 (uptrend) with volume > 1.8x average.
Short when price breaks below 20-period Donchian lower band and close < 12h EMA50 (downtrend) with volume > 1.8x average.
Exit when price crosses the 12h EMA50 (trend reversal) or hits ATR-based stoploss.
Uses 4h timeframe to target 75-200 total trades over 4 years. Donchian provides objective breakout levels.
Volume confirmation ensures breakout conviction. Trend filter prevents counter-trend trades.
Works in both bull and bear markets by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) on 12h for stoploss
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        dc_up = dc_upper[i]
        dc_low = dc_lower[i]
        ema50_val = ema50_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 12h EMA50 (uptrend) AND volume confirmation
            if (price > dc_up and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower AND price < 12h EMA50 (downtrend) AND volume confirmation
            elif (price < dc_low and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 12h EMA50 (trend reversal) OR ATR stoploss hit
                if price < ema50_val or price <= entry_price - 2.0 * atr_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 12h EMA50 (trend reversal) OR ATR stoploss hit
                if price > ema50_val or price >= entry_price + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0