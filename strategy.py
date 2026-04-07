#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(25) breakout with 1-hour volume confirmation and 4-hour KAMA trend filter
# Long when price breaks above 25-period Donchian high + volume > 2.0x 24-period average + KAMA rising
# Short when price breaks below 25-period Donchian low + volume > 2.0x 24-period average + KAMA falling
# Exit when price crosses 8-period EMA in opposite direction
# Stoploss at 2.5 * ATR(15)
# Position size: 0.28 (28% of capital)
# Uses 1-hour volume for confirmation and 4-hour KAMA for trend direction
# Target: 100-180 total trades over 4 years (25-45/year)

name = "4h_donchian25_1h_vol_4h_kama_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-hour data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 24:
        return np.zeros(n)
    
    # 4-hour data for KAMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 1-hour volume average (24-period)
    volume_1h = df_1h['volume'].values
    volume_1h_s = pd.Series(volume_1h)
    volume_ma = volume_1h_s.rolling(window=24, min_periods=24).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma)
    
    # Calculate 4-hour KAMA (30-period, fast=2, slow=30)
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.abs(np.diff(close_4h))
    volatility = np.concatenate([[volatility[0]], volatility])
    
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # 25-period Donchian channels
    highest_high = pd.Series(high).rolling(window=25, min_periods=25).max().values
    lowest_low = pd.Series(low).rolling(window=25, min_periods=25).min().values
    
    # 8-period EMA for exit
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # ATR(15) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(25, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(ema_8[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 8-period EMA
            elif close[i] < ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 8-period EMA
            elif close[i] > ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Look for entries: Donchian breakout with volume confirmation and KAMA trend filter
            # Volume filter: volume > 2.0x 24-period average
            volume_filter = volume[i] > 2.0 * volume_ma_aligned[i]
            # Trend filter: KAMA direction (rising for long, falling for short)
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            kama_falling = kama_aligned[i] < kama_aligned[i-1]
            
            # Long: price breaks above Donchian high + volume filter + KAMA rising
            if close[i] > highest_high[i] and volume_filter and kama_rising:
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + KAMA falling
            elif close[i] < lowest_low[i] and volume_filter and kama_falling:
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals