#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1w EMA50 > EMA200 with volume > 1.5x average.
Short when price breaks below lower Donchian channel and 1w EMA50 < EMA200 with volume > 1.5x average.
Exit on opposite Donchian break or EMA50/EMA200 crossover reversal.
Donchian channels provide structure-based breakouts in both bull and bear markets.
1w EMA50/EMA200 crossover filters for strong multi-week trend to avoid false breakouts in chop.
Volume confirmation ensures breakout legitimacy.
Designed for 1d timeframe targeting 30-100 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in direction of strong weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for EMA200
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 and EMA200 on 1w data
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema50_1w = calculate_ema(close_1w, 50)
    ema200_1w = calculate_ema(close_1w, 200)
    
    # Align 1w EMAs to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    def calculate_donchian(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_channel, lower_channel = calculate_donchian(high, low, 20)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        ema200_val = ema200_1w_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Determine weekly trend direction
        weekly_uptrend = ema50_val > ema200_val
        weekly_downtrend = ema50_val < ema200_val
        
        if position == 0:
            # Long: price breaks above upper Donchian AND weekly uptrend AND volume spike
            if (price > upper_val and weekly_uptrend and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND weekly downtrend AND volume spike
            elif (price < lower_val and weekly_downtrend and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR weekly trend turns down
                if (price < lower_val or not weekly_uptrend):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR weekly trend turns up
                if (price > upper_val or not weekly_downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_200_Trend_Volume"
timeframe = "1d"
leverage = 1.0