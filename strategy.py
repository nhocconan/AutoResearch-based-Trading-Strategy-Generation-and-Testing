#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation.
# Long when close > upper Donchian(20) AND price > 12h EMA50 AND volume > 1.5x 20-period volume MA.
# Short when close < lower Donchian(20) AND price < 12h EMA50 AND volume > 1.5x 20-period volume MA.
# Uses ATR-based stoploss: exit long when price < highest high since entry - 2.5*ATR,
# exit short when price < lowest low since entry + 2.5*ATR.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years.
# Donchian channels provide clear breakout levels, 12h EMA50 filters for trend direction,
# volume confirmation reduces false breakouts. Works in both bull and bear markets
# by only trading breakouts in the direction of the 12h trend.

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        ema12h_val = ema_50_12h_aligned[i]
        upper_donch = highest_high[i]
        lower_donch = lowest_low[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: close > upper Donchian AND price > 12h EMA50 AND volume spike
            if close_val > upper_donch and close_val > ema12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high_val
            # Short: close < lower Donchian AND price < 12h EMA50 AND volume spike
            elif close_val < lower_donch and close_val < ema12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low_val
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_val)
            
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price < highest high since entry - 2.5*ATR
            if close_val < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            # Exit: close < lower Donchian (breakdown)
            elif close_val < lower_donch:
                exit_signal = True
            # Exit: trend changes (price < 12h EMA50)
            elif close_val < ema12h_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price > lowest low since entry + 2.5*ATR
            if close_val > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            # Exit: close > upper Donchian (breakout)
            elif close_val > upper_donch:
                exit_signal = True
            # Exit: trend changes (price > 12h EMA50)
            elif close_val > ema12h_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals