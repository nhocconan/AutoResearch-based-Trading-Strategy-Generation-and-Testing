#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Camarilla levels provide statistically significant support/resistance levels.
# Breakouts above R4 or below S4 with volume confirmation indicate strong momentum.
# Weekly trend filter (price above/below 50-period EMA) ensures we trade with the higher timeframe trend.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (breakouts above R4 in uptrend) and bear markets (breakouts below S4 in downtrend).
name = "6h_Camarilla_R4S4_Breakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for indicators (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate previous day's Camarilla levels
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (using previous day's values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate weekly EMA50 for trend filter
    ema_period = 50
    close_1w = df_1w['close'].values
    ema_50 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        ema_50[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_50[i] = (close_1w[i] * (2/(ema_period+1))) + (ema_50[i-1] * (1 - 2/(ema_period+1)))
    
    # Align weekly EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R4 AND volume confirmation AND weekly uptrend (price > EMA50)
            long_breakout = close[i] > r4_aligned[i]
            weekly_uptrend = close[i] > ema_50_aligned[i]
            if vol_confirm and long_breakout and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND volume confirmation AND weekly downtrend (price < EMA50)
            elif vol_confirm and close[i] < s4_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R4 or weekly trend turns down
            exit_condition = close[i] < r4_aligned[i] or close[i] < ema_50_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S4 or weekly trend turns up
            exit_condition = close[i] > s4_aligned[i] or close[i] > ema_50_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals