#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Supertrend with weekly trend filter and volume confirmation
# Long when Supertrend turns green, weekly close > weekly EMA50 (uptrend), and volume > 1.5x 6h average volume
# Short when Supertrend turns red, weekly close < weekly EMA50 (downtrend), and volume > 1.5x 6h average volume
# Exit when Supertrend reverses or weekly trend changes
# Stoploss at 2.5 * ATR(10)
# Position size: 0.25 (25% of capital)
# Uses weekly EMA50 for trend filter and 6h volume average for confirmation
# Target: 80-160 total trades over 4 years (20-40/year)

name = "6h_supertrend_weekly_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Supertrend
    atr_period = 10
    multiplier = 3.0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full(n, np.nan)
    dir = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            dir[i] = 1
        elif close[i] < lower_band[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
        if dir[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # 6h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(supertrend[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Supertrend reverses or weekly trend changes
            elif dir[i] == -1 or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Supertrend reverses or weekly trend changes
            elif dir[i] == 1 or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Supertrend signal, weekly trend alignment, and volume confirmation
            # Long: Supertrend uptrend, weekly uptrend, volume spike
            if (dir[i] == 1 and
                close[i] > ema50_weekly_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Supertrend downtrend, weekly downtrend, volume spike
            elif (dir[i] == -1 and
                  close[i] < ema50_weekly_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals