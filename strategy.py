#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter. Elder Ray measures bull/bear power via EMA13.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. In bull regime (price > EMA50), enter long when Bull Power rises.
# In bear regime (price < EMA50), enter short when Bear Power rises. Trend filter on 1d EMA50 avoids whipsaw.
# Designed for low trade frequency (<40/year) to minimize fee bear markets.
name = "6h_ElderRay_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 1)  # Need 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_50_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        if position == 0:
            # Bull regime: price > 1d EMA50
            if close[i] > ema_1d:
                # Enter long when bull power is rising (current > previous)
                if i > 0 and bp > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Bear regime: price < 1d EMA50
            elif close[i] < ema_1d:
                # Enter short when bear power is rising (current > previous)
                if i > 0 and br > bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price < 1d EMA50 (trend change) OR bull power falls
            if close[i] < ema_1d or (i > 0 and bp < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 1d EMA50 (trend change) OR bear power falls
            if close[i] > ema_1d or (i > 0 and br < bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals