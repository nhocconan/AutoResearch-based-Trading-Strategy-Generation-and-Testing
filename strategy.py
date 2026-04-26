#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: On 6h timeframe, Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. 
Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
Go long when Bull Power > 0 and rising (bullish momentum) AND price > 1d EMA50 (uptrend filter).
Go short when Bear Power > 0 and rising (bearish momentum) AND price < 1d EMA50 (downtrend filter).
Use 1d EMA50 as regime filter to align with daily trend. 
ATR-based stoploss (2*ATR) manages risk. Discrete sizing (0.25) limits fee drag.
Works in bull markets (long when Bull Power strong) and bear markets (short when Bear Power strong).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray calculations (6h timeframe)
    ema_period = 13
    ema_13 = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = ema_13 - low   # Bear Power: EMA - Low
    
    # Slope of Bull/Bear Power (momentum confirmation)
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # ATR for stoploss (6h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA13 (13), EMA50 1d (50), ATR (14)
    start_idx = max(13, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or
            np.isnan(bear_power_slope[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        bull_power_slope_val = bull_power_slope[i]
        bear_power_slope_val = bear_power_slope[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bullish momentum) AND price > 1d EMA50 (uptrend)
            long_signal = (bull_power_val > 0) and (bull_power_slope_val > 0) and (close_val > ema_50_1d_val)
            
            # Short: Bear Power > 0 and rising (bearish momentum) AND price < 1d EMA50 (downtrend)
            short_signal = (bear_power_val > 0) and (bear_power_slope_val > 0) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power becomes negative OR ATR stoploss (2*ATR below entry)
            if (bull_power_val <= 0) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power becomes negative OR ATR stoploss (2*ATR above entry)
            if (bear_power_val <= 0) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0