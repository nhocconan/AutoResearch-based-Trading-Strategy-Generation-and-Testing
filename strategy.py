#!/usr/bin/env python3
name = "1d_Trend_Follow_With_Volume_Filter"
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
    
    # Daily EMA for trend following
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: up if weekly EMA20 rising, down if falling
        weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price above EMA50 in weekly uptrend with volume
            if close[i] > ema_50[i] and weekly_uptrend and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: price below EMA50 in weekly downtrend with volume
            elif close[i] < ema_50[i] and weekly_downtrend and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below EMA50 or weekly trend turns down
            if close[i] < ema_50[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above EMA50 or weekly trend turns up
            if close[i] > ema_50[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Daily trend following with weekly trend filter and volume confirmation
# - Uses daily EMA50 for trend identification on 1d chart
# - Weekly EMA20 acts as higher timeframe filter to ensure alignment with weekly trend
# - Volume confirmation (1.5x average) reduces false signals and whipsaws
# - Long when price > EMA50 AND weekly uptrend AND volume spike
# - Short when price < EMA50 AND weekly downtrend AND volume spike
# - Exit when price crosses back below/above EMA50 or weekly trend reverses
# - Position size 0.30 balances return potential with risk management
# - Designed to work in both bull and bear markets by following the weekly trend
# - Volume filter helps avoid choppy markets and false breakouts
# - Target: ~20-50 trades per year to stay within limits and minimize fee drag