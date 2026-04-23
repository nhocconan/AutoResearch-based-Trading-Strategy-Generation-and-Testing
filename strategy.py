#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high AND price > 1d EMA50 AND volume > 1.5x 20-period average volume.
Short when price breaks below 20-period low AND price < 1d EMA50 AND volume > 1.5x 20-period average volume.
Exit when price crosses 10-period moving average in opposite direction OR ATR trailing stop (2.0*ATR from extreme).
Donchian channels capture volatility-based breakouts; EMA50 filters for higher timeframe trend alignment; volume confirms breakout strength.
Works in both bull (breakouts up) and bear (breakouts down) markets by capturing momentum within the trend.
Target: ~15-25 trades/year on 12h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h 10-period MA for exit signal
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # ATR(14) for 12h trailing stop calculation
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
    start_idx = max(20, 50)  # donchian20, ema_50_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_val = highest_high[i]
        low_val = lowest_low[i]
        ema_val = ema_50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ma_val = ma_10[i]
        
        if position == 0:
            # Long: price breaks above 20-period high AND price > 1d EMA50 AND volume spike
            if price > high_val and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below 20-period low AND price < 1d EMA50 AND volume spike
            elif price < low_val and price < ema_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: price crosses 10-period MA in opposite direction
            if position == 1 and price < ma_val:
                exit_signal = True
            elif position == -1 and price > ma_val:
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

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_MA10Exit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0