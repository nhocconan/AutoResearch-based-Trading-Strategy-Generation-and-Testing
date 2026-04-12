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
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Use pandas rolling for proper min_periods handling
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume filter: 20-day EMA
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.5
        
        # Entry conditions: Donchian breakout with volume confirmation
        long_breakout = close[i] > donchian_high_daily[i]
        short_breakout = close[i] < donchian_low_daily[i]
        
        long_entry = long_breakout and volume_filter
        short_entry = short_breakout and volume_filter
        
        # Exit conditions: Opposite Donchian level touch
        long_exit = close[i] < donchian_low_daily[i]
        short_exit = close[i] > donchian_high_daily[i]
        
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

name = "1d_1w_donchian_breakout_volume_filter_v1"
timeframe = "1d"
leverage = 1.0