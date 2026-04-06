#!/usr/bin/env python3
"""
6h Elder Ray Index + 1w regime filter.
Hypothesis: Elder Ray (bull/bear power) captures institutional buying/selling pressure.
Weekly trend filter ensures we only take Elder Ray signals in the direction of the higher timeframe trend.
Volume confirmation adds confluence. Works in both bull (buy power > 0 in uptrend) and bear (bear power < 0 in downtrend).
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14315_6h_elder_ray_1w_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 13-period EMA for weekly trend (Elder Ray standard)
    ema_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 13-period EMA for 6x data (used in Elder Ray)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    ema_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_6h      # Bull Power = High - EMA(13)
    bear_power = low - ema_6h       # Bear Power = Low - EMA(13)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 13 for EMA, 20 for volume)
    start = max(13, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]) or np.isnan(bull_power[i]) or \
           np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: stoploss (2*ATR) or Elder Ray divergence
        if position == 1:  # long position
            if close[i] <= entry_price - 2.0 * atr[i] or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.0 * atr[i] or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with weekly trend and volume confirmation
            # Long when bull power > 0 in weekly uptrend with volume
            # Short when bear power < 0 in weekly downtrend with volume
            long_setup = (bull_power[i] > 0) and (close[i] > ema_1w_aligned[i]) and vol_confirm[i]
            short_setup = (bear_power[i] < 0) and (close[i] < ema_1w_aligned[i]) and vol_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals