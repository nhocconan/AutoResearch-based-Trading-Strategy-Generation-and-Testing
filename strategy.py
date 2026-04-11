#!/usr/bin/env python3
# 6h_1w_200ema_bull_bear_power_v1
# Strategy: 6h Bull/Bear Power with weekly EMA200 trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Bull/Bear Power (Elder Ray) measures bull/bear strength via high/low vs EMA13.
# Weekly EMA200 filters trend direction to avoid counter-trend trades. Volume confirms momentum.
# Works in bull via bull power > 0 above weekly EMA200, in bear via bear power < 0 below weekly EMA200.
# Low trade frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_200ema_bull_bear_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Weekly EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6h EMA(13) for Bull/Bear Power
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull/Bear Power components
    bull_power = high - ema_13  # High minus EMA13
    bear_power = low - ema_13   # Low minus EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bull/Bear Power conditions
        bull_strong = bull_power[i] > 0
        bear_weak = bear_power[i] < 0
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Bull/Bear Power + volume + trend alignment
        if bull_strong and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_weak and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Bull/Bear Power signal with volume confirmation
        elif position == 1 and (not bull_strong) and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bear_weak) and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals