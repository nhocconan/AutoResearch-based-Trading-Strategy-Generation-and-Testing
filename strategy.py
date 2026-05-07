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
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily Donchian breakout (20-period)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1d[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above 20-day high in weekly uptrend with volume
            if close[i] > donchian_high_20[i] and ema_10_1d[i] > ema_10_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low in weekly downtrend with volume
            elif close[i] < donchian_low_20[i] and ema_10_1d[i] < ema_10_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to 20-day low or weekly trend reverses
            if close[i] < donchian_low_20[i] or ema_10_1d[i] < ema_10_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to 20-day high or weekly trend reverses
            if close[i] > donchian_high_20[i] or ema_10_1d[i] > ema_10_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend following with daily Donchian breakouts and volume confirmation
# - Weekly EMA10 determines the trend direction (bullish when rising, bearish when falling)
# - Daily Donchian(20) breakouts provide entry signals in the direction of weekly trend
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price reverses to opposite Donchian band or weekly trend changes
# - Position size 0.25 balances risk and return while limiting trades
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Uses weekly timeframe for trend and daily for execution, avoiding overtrading
# - Target: 20-50 trades/year to minimize fee drag and improve generalization