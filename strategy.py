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
    
    # Get 4h data for Donchian channel and trend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_upper = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_lower = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume moving average (20-period)
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Calculate ATR for volatility filter (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 8-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma50[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8 * 20-day average volume
        volume_filter = volume[i] > (1.8 * volume_ma20_1d_aligned[i])
        
        # Volatility filter: ATR > 0.6 * 50-period ATR average (avoid low volatility chop)
        volatility_filter = atr[i] > (0.6 * atr_ma50[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper with trend, volume, and volatility
            if (close[i] > donchian_upper[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume_filter and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower against trend, volume, and volatility
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume_filter and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below Donchian lower or EMA34
            if close[i] < donchian_lower[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above Donchian upper or EMA34
            if close[i] > donchian_upper[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_EMA34_Volume_VolatilityFilter_Session"
timeframe = "4h"
leverage = 1.0