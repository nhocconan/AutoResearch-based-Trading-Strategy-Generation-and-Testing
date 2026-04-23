#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high, EMA50 > EMA200 (uptrend), and volume > 1.5x average.
Short when price breaks below 20-period Donchian low, EMA50 < EMA200 (downtrend), and volume > 1.5x average.
Exit when price crosses the 10-period EMA or ATR-based stoploss triggers.
Designed for moderate trade frequency (~25-50/year) to capture breakouts with trend alignment.
Works in bull markets via long breakouts and bear markets via short breakdowns.
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
    
    # Load 12-hour data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12-hour EMAs to 4-hour timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_12h, ema200)
    
    # Calculate ATR for stoploss (using 4-hour data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4-hour data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit signal
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        ema200_val = ema200_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        atr_val = atr[i]
        ema10_val = ema10[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Donchian breakout above, uptrend (EMA50 > EMA200), volume confirmation
            if (close_val > donchian_high_val and 
                ema50_val > ema200_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: Donchian breakdown below, downtrend (EMA50 < EMA200), volume confirmation
            elif (close_val < donchian_low_val and 
                  ema50_val < ema200_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        else:
            # Calculate stoploss levels
            if position == 1:
                # Long stoploss: entry price minus 2x ATR
                stop_loss = entry_price - 2.0 * atr_val
                # Exit conditions: price below 10 EMA OR stoploss hit
                exit_signal = (close_val < ema10_val) or (close_val <= stop_loss)
            else:  # position == -1
                # Short stoploss: entry price plus 2x ATR
                stop_loss = entry_price + 2.0 * atr_val
                # Exit conditions: price above 10 EMA OR stoploss hit
                exit_signal = (close_val > ema10_val) or (close_val >= stop_loss)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMATrend_Volume"
timeframe = "4h"
leverage = 1.0