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
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20) using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high/low (previous days only)
    high_20 = np.full(len(high_1d), np.nan)
    low_20 = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_1d, high_20)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume filter: 20-period EMA
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.5
        
        # Volatility filter: ATR > 0.3 * 20-period ATR mean
        atr_ma = np.full(n, np.nan)
        if i >= 34:
            atr_ma[i] = np.nanmean(atr[i-20:i])
        vol_filter = atr[i] > atr_ma[i] * 0.3 if not np.isnan(atr_ma[i]) else True
        
        # Entry conditions: Donchian breakout with volume and volatility
        long_entry = (close[i] > donch_high_4h[i]) and volume_filter and vol_filter
        short_entry = (close[i] < donch_low_4h[i]) and volume_filter and vol_filter
        
        # Exit conditions: Opposite Donchian level
        long_exit = close[i] < donch_low_4h[i]
        short_exit = close[i] > donch_high_4h[i]
        
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

name = "4h_1d_donchian_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0