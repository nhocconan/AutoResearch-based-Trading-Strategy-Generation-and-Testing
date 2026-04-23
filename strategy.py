#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and 12h EMA > price (uptrend) with volume > 1.5x average.
Short when price breaks below lower Donchian channel and 12h EMA < price (downtrend) with volume > 1.5x average.
Exit on opposite Donchian break or when 12h EMA crosses price (trend reversal).
Donchian channels provide clear structure, 12h EMA filters for higher timeframe trend alignment,
volume confirmation reduces false breakouts. Designed for 4h timeframe targeting 100-200 total trades over 4 years.
Works in both bull and bear markets by only taking breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h data
    def calculate_ema(values, period):
        ema = np.full_like(values, np.nan)
        if len(values) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema_50_12h = calculate_ema(close_12h, 50)
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on primary timeframe (4h)
    def calculate_donchian(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_period = 20
    upper_channel, lower_channel = calculate_donchian(high, low, donchian_period)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_50_12h_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 12h EMA > price (uptrend) AND volume spike
            if (price > upper_val and ema_val > price and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND 12h EMA < price (downtrend) AND volume spike
            elif (price < lower_val and ema_val < price and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR 12h EMA < price (trend reversal)
                if (price < lower_val or ema_val < price):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR 12h EMA > price (trend reversal)
                if (price > upper_val or ema_val > price):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0