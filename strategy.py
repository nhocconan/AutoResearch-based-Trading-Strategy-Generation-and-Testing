#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA filter.
Long when price breaks above 4h Donchian upper (20) and close > 1d EMA50 with volume > 1.5x average.
Short when price breaks below 4h Donchian lower (20) and close < 1d EMA50 with volume > 1.5x average.
Exit on opposite Donchian break or trend reversal. Uses 1h timeframe targeting 60-150 total trades over 4 years.
4h Donchian provides structural breakout levels, 1d EMA50 filters medium-term trend, volume confirms strength.
Designed to capture strong momentum moves while minimizing whipsaws via multi-timeframe confluence.
"""

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
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > donchian_upper and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below 4h Donchian lower AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < donchian_lower and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 4h Donchian lower OR trend reversal
                if (price < donchian_lower or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 4h Donchian upper OR trend reversal
                if (price > donchian_upper or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0