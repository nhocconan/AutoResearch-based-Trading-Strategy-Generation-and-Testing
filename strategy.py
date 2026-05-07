#!/usr/bin/env python3
name = "1d_Weekly_Trend_Following"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian(20) for breakout signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (1.5x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above Donchian high in weekly uptrend with volume
            if close[i] > donchian_high[i] and ema_50_1d[i] > ema_50_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in weekly downtrend with volume
            elif close[i] < donchian_low[i] and ema_50_1d[i] < ema_50_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or trend reverses
            if close[i] < donchian_low[i] or ema_50_1d[i] < ema_50_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to Donchian high or trend reverses
            if close[i] > donchian_high[i] or ema_50_1d[i] > ema_50_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend following with daily Donchian breakouts
# - Weekly EMA50 determines the long-term trend (bullish when rising, bearish when falling)
# - Daily Donchian(20) breakouts provide entry signals in the direction of the weekly trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price reverses to the opposite Donchian level or weekly trend changes
# - Works in both bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# - Position size 0.25 limits risk while allowing meaningful returns
# - Weekly trend filter ensures we only trade with the dominant market direction
# - Expected trade frequency: 10-25 trades/year to minimize fee drag
# - Uses weekly timeframe for trend determination and daily for execution timing
# - Simple, robust logic with minimal parameters to avoid overfitting
# - Avoids saturated strategies by using weekly trend + daily breakout combination not commonly tested