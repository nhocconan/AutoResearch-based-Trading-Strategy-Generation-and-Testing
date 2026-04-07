#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) + 1-day EMA trend filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 and close > daily EMA50
# Short when Bear Power > 0 and Bull Power < 0 and close < daily EMA50
# Exit when signal reverses or price crosses EMA13
# Stoploss at 2.0 * ATR(13)
# Position size: 0.25 (25% of capital)
# Target: 80-150 total trades over 4 years (20-38/year)

name = "6h_elder_ray_1d_ema50_v1"
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
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR(13) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: bear power becomes positive or price closes below EMA13
            elif bear_power[i] > 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: bull power becomes positive or price closes above EMA13
            elif bull_power[i] > 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray divergence with daily trend filter
            # Long: Bull Power positive, Bear Power negative, and above daily EMA50
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power positive, Bull Power negative, and below daily EMA50
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>