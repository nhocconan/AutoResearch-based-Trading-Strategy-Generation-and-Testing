#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Elder Ray with 12h trend filter and volatility filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and Bear Power < 0 and 12h EMA50 rising and ATR > 1.2*ATR20.
# Short when Bear Power > 0 and Bull Power < 0 and 12h EMA50 falling and ATR > 1.2*ATR20.
# Exit when Elder Ray signals reverse or ATR < 0.8*ATR20.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_elderray12h_ema50_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_prev = pd.Series(close_12h).ewm(span=50, adjust=False).mean().shift(1).values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_50_prev_aligned = align_htf_to_ltf(prices, df_12h, ema_50_prev)
    
    # EMA13 for Elder Ray (calculated on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_prev_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Elder Ray reverses OR volatility drops
            if bull_power[i] <= 0 or bear_power[i] >= 0 or atr[i] < 0.8 * atr_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Elder Ray reverses OR volatility drops
            if bear_power[i] <= 0 or bull_power[i] >= 0 or atr[i] < 0.8 * atr_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals + 12h EMA50 trend + volatility expansion
            if ema_50_aligned[i] > ema_50_prev_aligned[i] and atr[i] > 1.2 * atr_ma[i]:
                # Uptrend + volatility expansion: look for long
                if bull_power[i] > 0 and bear_power[i] < 0:
                    signals[i] = 0.25
                    position = 1
            elif ema_50_aligned[i] < ema_50_prev_aligned[i] and atr[i] > 1.2 * atr_ma[i]:
                # Downtrend + volatility expansion: look for short
                if bear_power[i] > 0 and bull_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
    
    return signals