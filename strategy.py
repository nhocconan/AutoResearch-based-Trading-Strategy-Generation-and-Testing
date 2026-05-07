#!/usr/bin/env python3
name = "1d_1wPivot_PriceAction_Volume"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly pivot points from weekly data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily price action: higher highs/lows for trend
    hh_condition = high > np.roll(high, 1)
    ll_condition = low > np.roll(low, 1)
    lh_condition = high < np.roll(high, 1)
    hl_condition = low < np.roll(low, 1)
    
    # Volume spike detection: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with higher low and volume in weekly uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            weekly_uptrend = weekly_close[i] > weekly_close[i-1] if i > 0 else False
            price_action = ll_condition[i] and hl_condition[i]  # higher low and higher high
            
            if close[i] > s1_aligned[i] and vol_condition and weekly_uptrend and price_action:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with lower high and volume in weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not weekly_uptrend and lh_condition[i] and hl_condition[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or trend breaks
            if close[i] < s1_aligned[i] or not ll_condition[i] or not hl_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or trend breaks
            if close[i] > r1_aligned[i] or not lh_condition[i] or not hl_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily price action with weekly pivot support/resistance
# - Weekly pivot points (S1/R1) act as key support/resistance levels
# - Long when price breaks above S1 with higher low/higher high structure and volume
# - Short when price breaks below R1 with lower high/lower low structure and volume
# - Volume confirmation (1.5x average) filters weak breakouts
# - Works in bull markets (buy S1 breaks in uptrend) and bear markets (sell R1 breaks in downtrend)
# - Exit when price returns to weekly pivot level or price action breaks down
# - Position size 0.25 targets 15-25 trades/year, minimizing fee drag
# - Weekly pivot provides structure that works across market regimes
# - Price action filters ensure we trade with momentum, not against it