#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 12-hour volume confirmation and 1-day ATR filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period 12h average + ATR(14) < 0.5 * 50-period ATR mean
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period 12h average + ATR(14) < 0.5 * 50-period ATR mean
# Exit when price crosses 6-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 12h volume for confirmation and daily ATR regime filter to avoid high volatility chop
# Target: 75-200 total trades over 4 years (19-50/year)

name = "6h_donchian20_12h_vol_1d_atr_regime_v1"
timeframe = "6h"
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
    
    # 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour volume average (20-period)
    volume_12h = df_12h['volume'].values
    volume_12h_s = pd.Series(volume_12h)
    volume_ma = volume_12h_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    # Calculate 1-day ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 50-period average of 1-day ATR for regime filter
    atr_1d_s = pd.Series(atr_1d)
    atr_ma_50 = atr_1d_s.rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6-period EMA for exit
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(ema_6[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ATR regime filter: only trade when current ATR < 50% of 50-day average ATR (low volatility regime)
        atr_filter = atr[i] < 0.5 * atr_ma_50_aligned[i]
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 6-period EMA
            elif close[i] < ema_6[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 6-period EMA
            elif close[i] > ema_6[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and ATR regime filter
            # Volume filter: volume > 1.5x 20-period 12h average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: price breaks above Donchian high + volume filter + ATR regime filter
            if close[i] > highest_high[i] and volume_filter and atr_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + ATR regime filter
            elif close[i] < lowest_low[i] and volume_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals