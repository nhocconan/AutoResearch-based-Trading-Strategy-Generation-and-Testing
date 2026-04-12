#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-day ATR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_14[i] = np.nanmean(tr[i-14:i+1])
    
    # Calculate 20-day volume average
    vol_20 = np.full(len(df_1d), np.nan)
    vol_series = pd.Series(volume_1d)
    vol_20_values = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_20[:] = vol_20_values
    
    # Align ATR and volume average to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_20_4h = align_htf_to_ltf(prices, df_1d, vol_20)
    
    # Calculate 4h ATR for volatility filter
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = tr2_4h[0] = tr3_4h[0] = np.nan
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.nanmean(tr_4h[i-14:i+1])
    
    # Calculate 4h 20-period ATR average for volatility regime
    atr_ma_20 = np.full(n, np.nan)
    atr_series = pd.Series(atr_4h)
    atr_ma_20_values = atr_series.rolling(window=20, min_periods=20).mean().values
    atr_ma_20[:] = atr_ma_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_14_4h[i]) or np.isnan(vol_20_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.5x 20-period ATR average
        vol_filter = atr_4h[i] > atr_ma_20[i] * 1.5
        
        # Volume filter: current volume > 1.5x 20-day average volume
        volume_filter = volume[i] > vol_20_4h[i] * 1.5
        
        # Entry conditions: High volatility + high volume (momentum breakout)
        # Long when price breaks above recent high with expansion
        # Short when price breaks below recent low with expansion
        high_20 = np.full(n, np.nan)
        low_20 = np.full(n, np.nan)
        if i >= 20:
            high_20[i] = np.max(high[i-20:i])
            low_20[i] = np.min(low[i-20:i])
        
        long_breakout = (not np.isnan(high_20[i])) and (close[i] > high_20[i])
        short_breakout = (not np.isnan(low_20[i])) and (close[i] < low_20[i])
        
        long_entry = long_breakout and vol_filter and volume_filter
        short_entry = short_breakout and vol_filter and volume_filter
        
        # Exit conditions: Volatility contraction or opposite breakout
        long_exit = (not np.isnan(low_20[i])) and (close[i] < low_20[i]) or (atr_4h[i] < atr_ma_20[i] * 0.8)
        short_exit = (not np.isnan(high_20[i])) and (close[i] > high_20[i]) or (atr_4h[i] < atr_ma_20[i] * 0.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_volatility_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0