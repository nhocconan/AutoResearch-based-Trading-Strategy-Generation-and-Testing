#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Spike
Hypothesis: Donchian channel breakout on 12h for entry, filtered by volume spike and 1-week trend filter.
In bull markets: follow long breakouts when 1w trend is up. In bear markets: follow short breakouts when 1w trend is down.
Uses 12h for entry timing, 1w for trend filter to reduce whipsaw and false signals.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25 to manage drawdown.
"""

name = "12h_Donchian20_Breakout_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return ema
        multiplier = 2.0 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema34_1w = calculate_ema(close_1w, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 12-period Donchian channels
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume spike (volume > 2.0 * 20-period volume average)
    def calculate_volume_spike(volume, period=20):
        vol_ma = np.full_like(volume, np.nan)
        for i in range(period-1, len(volume)):
            vol_ma[i] = np.mean(volume[i-(period-1):i+1])
        spike = np.zeros_like(volume)
        for i in range(len(volume)):
            if not np.isnan(vol_ma[i]) and volume[i] > 2.0 * vol_ma[i]:
                spike[i] = 1.0
        return spike
    
    vol_spike = calculate_volume_spike(volume, 20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_spike[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper + volume spike + 1w trend up (close > EMA34)
            if close[i] > donchian_upper[i] and vol_spike[i] > 0 and close_1w[-1] > ema34_1w[-1] if len(close_1w) > 0 else False:
                # Use aligned 1w trend for current bar
                if not np.isnan(ema34_1w_aligned[i]) and close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: Price breaks below Donchian lower + volume spike + 1w trend down (close < EMA34)
            elif close[i] < donchian_lower[i] and vol_spike[i] > 0 and close_1w[-1] < ema34_1w[-1] if len(close_1w) > 0 else False:
                if not np.isnan(ema34_1w_aligned[i]) and close[i] < ema34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian lower OR 1w trend turns down
            if close[i] < donchian_lower[i] or (not np.isnan(ema34_1w_aligned[i]) and close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian upper OR 1w trend turns up
            if close[i] > donchian_upper[i] or (not np.isnan(ema34_1w_aligned[i]) and close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals