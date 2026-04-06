#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h EMA(50) filter and ATR stop.
# Elder Ray Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Go long when Bull Power > 0 and Bear Power < 0 (bulls in control)
# Go short when Bear Power < 0 and Bull Power < 0 (bears in control)
# Use 12h EMA(50) to filter trades: only long when 12h EMA(50) rising, short when falling
# Exit on opposite signal or when price moves 2*ATR against position
# Target: 100-200 total trades over 4 years (25-50/year)

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
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Elder Ray components: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = low - ema_13   # Low - EMA(13)
    
    # ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: reverse signal OR stop loss (2*ATR)
            if (bull_power[i] <= 0 and bear_power[i] >= 0) or \
               (close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reverse signal OR stop loss (2*ATR)
            if (bull_power[i] >= 0 and bear_power[i] <= 0) or \
               (close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray + 12h EMA(50) trend filter
            ema_rising = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            ema_falling = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
            
            # Long: bulls in control (bull power > 0, bear power < 0) + rising 12h EMA
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_rising:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bears in control (bear power < 0, bull power < 0) + falling 12h EMA
            elif bear_power[i] < 0 and bull_power[i] < 0 and ema_falling:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals