#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA(50) trend filter and ATR(14) volatility filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bull Power > 0, price > EMA50, and ATR > 0.5 * ATR mean (volatility filter).
# Enter short when Bear Power > 0, price < EMA50, and ATR > 0.5 * ATR mean.
# Exit when opposite signal occurs or ATR drops below threshold.
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
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_mean = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_threshold = 0.5 * atr_mean  # Require ATR > 50% of its 50-period mean
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bear Power > 0 (bearish pressure) OR ATR drops below threshold
            if bear_power[i] > 0 or atr[i] < atr_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power > 0 (bullish pressure) OR ATR drops below threshold
            if bull_power[i] > 0 or atr[i] < atr_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray + EMA50 trend + ATR volatility filter
            if atr[i] > atr_threshold[i]:
                if bull_power[i] > 0 and close[i] > ema_50_aligned[i]:
                    # Bullish pressure + uptrend + sufficient volatility: long
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] > 0 and close[i] < ema_50_aligned[i]:
                    # Bearish pressure + downtrend + sufficient volatility: short
                    signals[i] = -0.25
                    position = -1
    
    return signals