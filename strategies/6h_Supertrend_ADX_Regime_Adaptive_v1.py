#!/usr/bin/env python3
"""
6h_Supertrend_ADX_Regime_Adaptive_v1
Hypothesis: 6h Supertrend with ADX regime filter and dynamic position sizing. In trending markets (ADX>25), follow Supertrend direction. In ranging markets (ADX<20), mean revert at Supertrend extremes. Uses 1d HTF trend alignment for higher timeframe context. Targets 50-150 trades over 4 years by requiring regime alignment and trend confirmation. Works in bull/bear via adaptive logic: trend following in strong trends, mean reversion in chop. Discrete position sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for Supertrend (using 10-period ATR, multiplier 3.0)
    atr_period = 10
    atr_mult = 3.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (atr_mult * atr)
    lowerband = hl2 - (atr_mult * atr)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
            direction[i] = -1
    
    # ADX calculation for regime filtering
    adx_period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr_14 = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 14 for ADX/Supertrend)
    start_idx = max(50, adx_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(supertrend[i]) or
            np.isnan(direction[i]) or np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime-based logic
        if adx[i] > 25:  # Trending regime
            # Follow Supertrend direction aligned with HTF trend
            if direction[i] == 1 and htf_trend[i] == 1:  # Uptrend alignment
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif direction[i] == -1 and htf_trend[i] == -1:  # Downtrend alignment
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Regime mismatch - exit
                signals[i] = 0.0
                position = 0
        elif adx[i] < 20:  # Ranging regime
            # Mean revert at Supertrend extremes
            if close[i] < lowerband[i] and htf_trend[i] == 1:  # Long mean reversion in uptrend HTF
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] > upperband[i] and htf_trend[i] == -1:  # Short mean reversion in downtrend HTF
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion position when price returns to Supertrend
                if position == 1 and close[i] > supertrend[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] < supertrend[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
        else:  # Transition regime (20 <= ADX <= 25)
            # Hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Supertrend_ADX_Regime_Adaptive_v1"
timeframe = "6h"
leverage = 1.0