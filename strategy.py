#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with weekly trend filter and ATR-based stops.
# Elder Ray measures bull/bear power via EMA(13) deviation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Weekly trend filter (price above/below 20-week EMA) ensures alignment with higher timeframe trend.
# ATR stoploss limits drawdown. Works in bull via bull power strength and bear via bear power weakness.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_elder_ray_trend_filter_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray: EMA(13) and Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: 20-week EMA on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        if i == 19:
            ema_20w[i] = np.mean(close_1w[0:20])
        else:
            ema_20w[i] = close_1w[i] * 2/(20+1) + ema_20w[i-1] * (1 - 2/(20+1))
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_20w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power turns negative or stoploss
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if (bear_power[i] < 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns positive or stoploss
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if (bull_power[i] > 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter
            # Long: bull power positive and above weekly uptrend
            if (bull_power[i] > 0 and 
                close[i] > ema_20w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bear power negative and below weekly downtrend
            elif (bear_power[i] < 0 and 
                  close[i] < ema_20w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals