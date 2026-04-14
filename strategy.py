#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bull/Bear Power with 1-day EMA filter
# Elder Ray's Bull/Bear Power: measures buying/selling pressure relative to EMA
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power rising AND price > EMA(50) from 1d
# Short when Bear Power < 0 AND Bull Power falling AND price < EMA(50) from 1d
# Exit when power crosses zero or price crosses EMA(50)
# Works in both bull and bear markets by measuring institutional strength
# Target: 80-160 total trades over 4 years (20-40/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(13) for Bull/Bear Power (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema13
    
    # Calculate EMA(50) from 1d for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean()
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        bep = bear_power[i]
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long setup: Bull Power positive AND Bear Power rising AND price above 1d EMA50
            if (bp > 0 and 
                i > start and bep > bear_power[i-1] and 
                price > ema50):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power negative AND Bull Power falling AND price below 1d EMA50
            elif (bep < 0 and 
                  i > start and bp < bull_power[i-1] and 
                  price < ema50):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative OR price below 1d EMA50
            if bp <= 0 or price < ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive OR price above 1d EMA50
            if bep >= 0 or price > ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_BullBearPower_1dEMA50_Filter"
timeframe = "6h"
leverage = 1.0