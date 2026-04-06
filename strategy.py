#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(25) trend filter and volume confirmation.
# Enter long when price breaks above Donchian(20) high AND 12h EMA25 rising AND volume > 1.5x average.
# Enter short when price breaks below Donchian(20) low AND 12h EMA25 falling AND volume > 1.5x average.
# Exit on opposite Donchian break or when price crosses 12h EMA25.
# Uses ATR(10) stoploss: exit if price moves 2*ATR against position.
# Designed to capture trends in both bull and bear markets with filtered entries to avoid overtrading.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_donchian20_12h_ema25_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(25) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_25 = pd.Series(close_12h).ewm(span=25, adjust=False).mean().values
    ema_25_prev = np.roll(ema_25, 1)
    ema_25_prev[0] = ema_25[0]
    ema_25_rising = ema_25 > ema_25_prev
    ema_25_falling = ema_25 < ema_25_prev
    ema_25_aligned = align_htf_to_ltf(prices, df_12h, ema_25)
    ema_25_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_25_rising)
    ema_25_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_25_falling)
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg)
    
    # ATR(10) for stoploss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_25_aligned[i]) or np.isnan(ema_25_rising_aligned[i]) or 
            np.isnan(ema_25_falling_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_confirm[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian low OR price below EMA25 OR stoploss hit
            if (close[i] < low_roll[i] or close[i] < ema_25_aligned[i] or 
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high OR price above EMA25 OR stoploss hit
            if (close[i] > high_roll[i] or close[i] > ema_25_aligned[i] or 
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + EMA25 trend + volume confirmation
            if vol_confirm[i]:
                if close[i] > high_roll[i] and ema_25_rising_aligned[i]:
                    # Breakout above Donchian high with rising EMA25: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_roll[i] and ema_25_falling_aligned[i]:
                    # Breakdown below Donchian low with falling EMA25: short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals