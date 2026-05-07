#!/usr/bin/env python3
name = "1d_Donchian_Trend_Pullback"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Daily EMA for pullback entries
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: above average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: breakout above Donchian high in weekly uptrend on pullback
            if (close[i] > donch_high[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # weekly uptrend
                close[i] > ema_50[i] and vol_ok):  # pullback to EMA50
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in weekly downtrend on pullback
            elif (close[i] < donch_low[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # weekly downtrend
                  close[i] < ema_50[i] and vol_ok):  # pullback to EMA50
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA50 or breakdown
            if close[i] < ema_50[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA50 or breakout
            if close[i] > ema_50[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakouts with weekly EMA trend filter and EMA50 pullback entries
# - Long: Price breaks above 20-day Donchian high during weekly uptrend, then pulls back to EMA50
# - Short: Price breaks below 20-day Donchian low during weekly downtrend, then pulls back to EMA50
# - Weekly EMA50 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Pullback to EMA50 provides better risk-reward than chasing breakouts
# - Exit when price returns to EMA50 or breaks opposite Donchian band
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Works in bull markets (long breakouts in uptrend) and bear markets (short breakdowns in downtrend)
# - Donchian channels provide objective breakout levels with clear invalidation
# - Weekly trend filter reduces whipsaws vs same-timeframe breakout strategies
# - Combines trend following (Donchian breakout) with value entry (pullback to EMA)