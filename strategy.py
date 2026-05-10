#!/usr/bin/env python3
# 6h_ElderRay_MeanReversion_1dTrend
# Hypothesis: Elder Ray (Bull Power/Bear Power) with 1-day trend filter and mean reversion at extreme levels.
# In strong 1-day trends, price often reverts to the 13-period EMA after extended moves.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bear Power is extremely negative (oversold) in uptrend.
# Enter short when Bull Power is extremely high (overbought) in downtrend.
# Uses 1-day EMA50 for trend alignment and 13-period EMA for Elder Ray calculation.
# Targets 50-150 trades over 4 years via extreme readings and trend alignment.

name = "6h_ElderRay_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray indicators
    bull_power = high - ema_13  # Higher = more bullish
    bear_power = ema_13 - low   # Higher = more bearish (since it's EMA13 - Low)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Normalize Elder Ray by ATR(20) for adaptive thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Avoid division by zero
    atr_20_safe = np.where(atr_20 == 0, 1e-10, atr_20)
    bull_power_norm = bull_power / atr_20_safe
    bear_power_norm = bear_power / atr_20_safe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for EMA13, ATR20, and 1d EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_power_norm[i]) or
            np.isnan(bear_power_norm[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Extreme levels for mean reversion (adaptive thresholds)
        # Enter long when bear power is extremely negative (oversold) in uptrend
        # Enter short when bull power is extremely high (overbought) in downtrend
        if position == 0:
            # Long signal: extreme bear power (oversold) in uptrend
            if bear_power_norm[i] < -2.0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short signal: extreme bull power (overbought) in downtrend
            elif bull_power_norm[i] > 2.0 and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: bear power normalizes or trend fails
                if bear_power_norm[i] > -0.5 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: bull power normalizes or trend fails
                if bull_power_norm[i] < 0.5 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals