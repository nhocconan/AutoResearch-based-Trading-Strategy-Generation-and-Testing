#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1w EMA50 is rising with volume > 1.5x average.
Short when price breaks below lower Donchian channel and 1w EMA50 is falling with volume > 1.5x average.
Exit on opposite Donchian break or EMA50 direction change.
Donchian channels provide clear trend-following structure.
1w EMA50 > 1w EMA200 filters for strong long-term trend to avoid false breakouts in chop.
Volume confirmation ensures breakout legitimacy.
Designed for 1d timeframe targeting 30-100 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in direction of strong long-term trend.
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
    
    # Load 1w data for EMA50/EMA200 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 and EMA200 on 1w data
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema50_1w = calculate_ema(close_1w, 50)
    ema200_1w = calculate_ema(close_1w, 200)
    
    # Align 1w EMA50/EMA200 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Donchian(20) channels from prior 1d bar
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_donch = np.full(len(close), np.nan)
    lower_donch = np.full(len(close), np.nan)
    
    for i in range(20-1, len(close)):
        upper_donch[i] = np.max(high[i-20+1:i+1])
        lower_donch[i] = np.min(low[i-20+1:i+1])
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        ema200_val = ema200_1w_aligned[i]
        upper_val = upper_donch[i]
        lower_val = lower_donch[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Determine 1w trend: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        is_uptrend = ema50_val > ema200_val
        is_downtrend = ema50_val < ema200_val
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1w uptrend AND volume spike
            if (price > upper_val and is_uptrend and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND 1w downtrend AND volume spike
            elif (price < lower_val and is_downtrend and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR 1w trend turns down
                if (price < lower_val or not is_uptrend):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR 1w trend turns up
                if (price > upper_val or not is_downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0