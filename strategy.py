#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses 4h Donchian(20) breakout with volume > 1.5x 24-period average
# Enters only when 1d EMA(50) > EMA(200) for longs or < for shorts (trend filter)
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "4h_1d_donchian_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMAs for trend filter
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(df_1d), np.nan)
    ema_200 = np.full(len(df_1d), np.nan)
    
    # EMA calculation with proper smoothing
    alpha_50 = 2 / (50 + 1)
    alpha_200 = 2 / (200 + 1)
    
    # Initialize EMAs
    ema_50[0] = close_1d[0]
    ema_200[0] = close_1d[0]
    
    for i in range(1, len(df_1d)):
        ema_50[i] = alpha_50 * close_1d[i] + (1 - alpha_50) * ema_50[i-1]
        ema_200[i] = alpha_200 * close_1d[i] + (1 - alpha_200) * ema_200[i-1]
    
    # Trend: 1 if EMA50 > EMA200, -1 if EMA50 < EMA200, 0 otherwise
    trend_1d = np.zeros(len(df_1d))
    trend_1d[ema_50 > ema_200] = 1
    trend_1d[ema_50 < ema_200] = -1
    
    # Align trend to 4h timeframe
    trend_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 4h Donchian channel (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 24-period average on 4h (6 days)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(trend_4h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low
            if close[i] <= donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high
            if close[i] >= donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian high with volume confirmation and uptrend
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] > donch_high[i] and 
                vol_ratio > 1.5 and 
                trend_4h[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian low with volume confirmation and downtrend
            elif (close[i] < donch_low[i] and 
                  vol_ratio > 1.5 and 
                  trend_4h[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals