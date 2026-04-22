#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + weekly trend filter
# Elder Ray (Bull/Bear power) measures market momentum strength.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Strong Bull Power (>0.5*ATR) + weekly uptrend → Long
# Strong Bear Power (>0.5*ATR) + weekly downtrend → Short
# Weekly trend uses 200 EMA to filter counter-trend trades.
# Targets 15-25 trades/year with controlled risk.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(200) for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h ATR(14) for power threshold
    high_low = high - low
    high_close_prev = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_prev = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(atr[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power + weekly uptrend
            if (bull_power[i] > 0.5 * atr[i] and  # strong bullish momentum
                close[i] > ema_200_1w_aligned[i]):  # weekly uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power + weekly downtrend
            elif (bear_power[i] > 0.5 * atr[i] and  # strong bearish momentum
                  close[i] < ema_200_1w_aligned[i]):  # weekly downtrend filter
                signals[i] = -0.25
                position = -1
        else:
            # Exit: momentum weakens or trend reverses
            if position == 1:
                # Exit long: weak bull power or weekly trend turns down
                if (bull_power[i] < 0.2 * atr[i] or 
                    close[i] < ema_200_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: weak bear power or weekly trend turns up
                if (bear_power[i] < 0.2 * atr[i] or 
                    close[i] > ema_200_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0