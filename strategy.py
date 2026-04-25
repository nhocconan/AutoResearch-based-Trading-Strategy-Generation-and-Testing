#!/usr/bin/env python3
"""
6h_AsymmetricRegime_ADX_EMA21_v1
Hypothesis: Use asymmetric logic per regime - ADX>25 + price<SMA50 = bear regime (only short retrace to EMA21); ADX<20 = range regime (mean revert at Bollinger Bands); hysteresis prevents whipsaw. Works in bull (buy range dips) and bear (short bear retracements).
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ADX(14)
    plus_dm = high_s.diff()
    minus_dm = low_s.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    
    # SMA(50) and EMA(21)
    sma_50 = close_s.rolling(window=50, min_periods=50).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bollinger Bands(20,2)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean()
    std_20 = close_s.rolling(window=20, min_periods=20).std()
    upper_bb = (sma_20 + 2 * std_20).values
    lower_bb = (sma_20 - 2 * std_20).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    regime = 0    # 0: unknown, 1: bull, -1: bear, 2: range
    
    # Start index: need warmup for SMA50 and ADX
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_values[i]) or np.isnan(sma_50[i]) or np.isnan(ema_21[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime with hysteresis
        if regime == 0:  # initial regime detection
            if adx_values[i] > 25 and close[i] < sma_50[i]:
                regime = -1  # bear
            elif adx_values[i] < 20:
                regime = 2   # range
            else:
                regime = 1   # bull (default when not clearly bear or range)
        elif regime == -1:  # bear regime
            if adx_values[i] < 18:  # hysteresis exit
                regime = 2
            elif adx_values[i] > 25 and close[i] >= sma_50[i]:
                regime = 1
        elif regime == 2:   # range regime
            if adx_values[i] > 22 and close[i] < sma_50[i]:
                regime = -1  # bear
            elif adx_values[i] > 22 and close[i] >= sma_50[i]:
                regime = 1   # bull
        elif regime == 1:   # bull regime
            if adx_values[i] < 18:  # hysteresis exit
                regime = 2
            elif adx_values[i] > 25 and close[i] < sma_50[i]:
                regime = -1  # bear
        
        if position == 0:
            if regime == -1:  # bear: short retrace to EMA21
                if close[i] > ema_21[i]:
                    signals[i] = -0.25
                    position = -1
            elif regime == 2:   # range: mean revert at BB
                if close[i] <= lower_bb[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb[i]:
                    signals[i] = -0.25
                    position = -1
            elif regime == 1:   # bull: buy range dips
                if close[i] <= lower_bb[i]:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            signals[i] = 0.25
            if regime == -1:    # bear: exit long if trend turns bear
                if adx_values[i] > 25 and close[i] < sma_50[i]:
                    signals[i] = 0.0
                    position = 0
            elif regime == 2:   # range: exit at mean or opposite BB
                if close[i] >= sma_20.iloc[i] or close[i] >= upper_bb[i]:
                    signals[i] = 0.0
                    position = 0
            elif regime == 1:   # bull: exit at upper BB or trend change
                if close[i] >= upper_bb[i] or (adx_values[i] > 25 and close[i] < sma_50[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            signals[i] = -0.25
            if regime == -1:    # bear: exit at mean or lower BB
                if close[i] <= sma_20.iloc[i] or close[i] <= lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
            elif regime == 2:   # range: exit at mean or opposite BB
                if close[i] <= sma_20.iloc[i] or close[i] <= lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
            elif regime == 1:   # bull: exit short if trend turns bull
                if adx_values[i] > 25 and close[i] >= sma_50[i]:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_AsymmetricRegime_ADX_EMA21_v1"
timeframe = "6h"
leverage = 1.0