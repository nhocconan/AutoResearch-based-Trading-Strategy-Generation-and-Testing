#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w trend filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) and price > 1w EMA34 (uptrend)
# Short when Bull Power < 0 and Bear Power > 0 (bearish momentum) and price < 1w EMA34 (downtrend)
# Exit when momentum diverges (Bull Power <= 0 for longs, Bear Power <= 0 for shorts)
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.
# Elder Ray measures bull/bear power relative to EMA, works in both bull (strong momentum) and bear (failed rallies via exits) markets.
# 1w EMA34 filter ensures we only trade with the higher timeframe trend, reducing whipsaw.

name = "6h_ElderRay_1wEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Elder Ray components: EMA13 of close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions with 1w trend filter
        long_entry = bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_34_1w_aligned[i]
        short_entry = bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema_34_1w_aligned[i]
        
        # Exit conditions: momentum divergence
        long_exit = bull_power[i] <= 0  # Long exit when bull power fades
        short_exit = bear_power[i] <= 0  # Short exit when bear power fades
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals