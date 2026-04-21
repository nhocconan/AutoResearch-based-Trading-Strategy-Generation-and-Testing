#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-week ATR-based volatility regime filter combined with 1-day EMA trend and 12h price momentum.
Long when: price > daily EMA34, 12h momentum > 0, and weekly ATR contraction (volatility decreasing).
Short when: price < daily EMA34, 12h momentum < 0, and weekly ATR contraction.
Volatility regime filter reduces whipsaws in choppy markets. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Load 1w data for ATR-based volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR(14) on weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly ATR contraction: current ATR < ATR 2 weeks ago
    atr_contraction = np.zeros(len(atr_1w), dtype=bool)
    atr_contraction[14:] = atr_1w[14:] < atr_1w[:-14]
    atr_contraction_aligned = align_htf_to_ltf(prices, df_1w, atr_contraction.astype(float))
    
    # 12h price momentum (rate of change over 3 periods)
    roc_period = 3
    close_12h = prices['close'].values
    roc = np.zeros_like(close_12h)
    roc[roc_period:] = (close_12h[roc_period:] - close_12h[:-roc_period]) / close_12h[:-roc_period]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr_contraction_aligned[i]) or 
            np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_12h[i]
        ema_trend = ema_34_aligned[i]
        vol_regime = atr_contraction_aligned[i] > 0.5  # True if volatility contracting
        momentum = roc[i]
        
        if position == 0:
            # Enter long: uptrend + positive momentum + volatility contraction
            if (price_close > ema_trend and 
                momentum > 0 and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + negative momentum + volatility contraction
            elif (price_close < ema_trend and 
                  momentum < 0 and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR momentum divergence
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # Momentum divergence exit (optional early exit)
            if position == 1 and momentum < -0.005:  # Strong negative momentum
                exit_signal = True
            elif position == -1 and momentum > 0.005:  # Strong positive momentum
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_VolatilityRegime_EMA34_Momentum"
timeframe = "12h"
leverage = 1.0