#!/usr/bin/env python3
name = "1d_Weekly_Trend_Following_With_Volume"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above Donchian high in weekly uptrend with volume
            if close[i] > donchian_high[i] and ema_20_1d[i] > ema_20_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in weekly downtrend with volume
            elif close[i] < donchian_low[i] and ema_20_1d[i] < ema_20_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or weekly trend reverses
            if close[i] < donchian_low[i] or ema_20_1d[i] < ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to Donchian high or weekly trend reverses
            if close[i] > donchian_high[i] or ema_20_1d[i] > ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend filter with daily Donchian breakout and volume confirmation
# - Weekly EMA20 determines the trend direction (only trade in trend direction)
# - Daily Donchian(20) breakout provides entry signals in the direction of weekly trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to opposite Donchian level or weekly trend reverses
# - Position size 0.25 balances risk and return while keeping trade frequency low
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Target: ~15-30 trades/year to avoid excessive fee drag
# - Uses weekly timeframe for trend filter and daily for execution timing
# - Simple 3-condition strategy reduces overfitting and improves robustness