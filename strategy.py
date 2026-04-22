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
    
    # Load 1d data for Donchian channels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channel (20-day high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period rolling max/min for Donchian channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: 50-period average on 4h
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_50[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high + volume spike + ATR filter
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_avg_50[i] and
                atr[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low + volume spike + ATR filter
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_avg_50[i] and
                  atr[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to the opposite Donchian level
            if position == 1:
                # Exit long: Price closes below daily Donchian low
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above daily Donchian high
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_DailyDonchian20_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0