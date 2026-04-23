#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h EMA trend filter + volume spike + Donchian(20) breakout for entries.
Long when price breaks above 20-period Donchian high AND 12h EMA(50) is rising AND volume > 2.0x 20-period average.
Short when price breaks below 20-period Donchian low AND 12h EMA(50) is falling AND volume > 2.0x 20-period average.
Exit via ATR trailing stop (3.0*ATR from highest/lowest since entry) or time-based exit (hold max 12 bars).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Combines trend (12h EMA), momentum (Donchian breakout), and volume confirmation to capture strong moves
while filtering choppy markets. Works in both bull and bear markets by requiring aligned trend and volume.
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
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian needs 20, 12h EMA needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_12h_aligned[i]
        ema_prev = ema_12h_aligned[i-1] if i > 0 else ema_trend
        
        # Determine if 12h EMA is rising/falling
        ema_rising = ema_trend > ema_prev
        ema_falling = ema_trend < ema_prev
        
        if position == 0:
            # Long: Donchian breakout + rising 12h EMA + volume spike
            if (price > donch_high and ema_rising and volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                bars_since_entry = 0
            # Short: Donchian breakdown + falling 12h EMA + volume spike
            elif (price < donch_low and ema_falling and volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                bars_since_entry = 0
        else:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # ATR-based trailing stop: 3.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 3.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 3.0 * atr_val:
                exit_signal = True
            
            # Time-based exit: hold max 12 bars (3 days on 4h)
            if bars_since_entry >= 12:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_12hEMA50Trend_VolumeSpike_ATRTrailingStop_TimeExit"
timeframe = "4h"
leverage = 1.0