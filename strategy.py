#!/usr/bin/env python3
name = "1d_Donchian20_WeeklyTrend_VolumeFilter"
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
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1d[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above upper band in weekly uptrend with volume
            if close[i] > high_roll[i] and ema_20_1d[i] > ema_20_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in weekly downtrend with volume
            elif close[i] < low_roll[i] and ema_20_1d[i] < ema_20_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to lower band or trend reverses
            if close[i] < low_roll[i] or ema_20_1d[i] < ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to upper band or trend reverses
            if close[i] > high_roll[i] or ema_20_1d[i] > ema_20_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on daily timeframe with weekly trend filter and volume confirmation
# - Buy when price breaks above 20-day high in weekly uptrend (EMA20 rising) with volume spike
# - Sell when price breaks below 20-day low in weekly downtrend (EMA20 falling) with volume spike
# - Exit when price returns to opposite band or weekly trend reverses
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# - Volume filter reduces false breakouts; weekly trend ensures alignment with higher timeframe momentum
# - Position size 0.25 balances return and risk, targeting ~15-25 trades/year to minimize fee drag
# - Weekly trend filter avoids counter-trend trades, improving win rate in both bull and bear regimes