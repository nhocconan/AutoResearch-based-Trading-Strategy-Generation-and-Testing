#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 13-period EMA and 1d trend filter.
# Elder Ray Power = Close - EMA13 (Bull Power) and EMA13 - Close (Bear Power).
# Long when Bull Power > 0 and rising, price above 1d EMA50.
# Short when Bear Power > 0 and rising, price below 1d EMA50.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Designed for both bull and bear markets by filtering with higher timeframe trend.
name = "6h_ElderRay_EMA13_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = EMA13 - Close
    bear_power = ema13 - close
    
    # Rising Bull Power: current > previous
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bull_power_rising[0] = False
    
    # Rising Bear Power: current > previous
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bear_power_rising[0] = False
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Bull Power positive and rising, price above 1d EMA50
            if bull_power[i] > 0 and bull_power_rising[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive and rising, price below 1d EMA50
            elif bear_power[i] > 0 and bear_power_rising[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power becomes negative or stops rising
            if bull_power[i] <= 0 or not bull_power_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power becomes negative or stops rising
            if bear_power[i] <= 0 or not bear_power_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals