#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x average + ADX > 25
# Short when price breaks below 20-period Donchian low + volume > 1.5x average + ADX > 25
# Uses 1d ADX for trend strength to avoid false breakouts in ranging markets
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets

name = "12h_donchian20_1d_vol_adx_v1"
timeframe = "12h"
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
    
    # 1d data for volume average and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d data to 12h timeframe
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(vol_avg_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or trend weakens
            elif close[i] < donchian_low[i] or adx_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or trend weakens
            elif close[i] > donchian_high[i] or adx_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above Donchian high + volume > 1.5x average + ADX > 25
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_avg_1d_aligned[i] and
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume > 1.5x average + ADX > 25
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_avg_1d_aligned[i] and
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals