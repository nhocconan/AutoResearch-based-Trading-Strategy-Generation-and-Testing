#!/usr/bin/env python3
"""
12h_RSI_Reversal_with_1dTrend_Filter
Hypothesis: RSI extremes combined with daily trend filter captures mean-reversion
in ranging markets while avoiding counter-trend moves in strong trends.
Works in both bull and bear markets by filtering trades with 1-day EMA trend.
Target: 20-40 trades per year (~80-160 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_RSI_Reversal_with_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) on 12h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in uptrend regime (buy dips in uptrend)
            # Short: RSI overbought (>70) in downtrend regime (sell rallies in downtrend)
            long_entry = (rsi[i] < 30) and uptrend_regime
            short_entry = (rsi[i] > 70) and downtrend_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend turns down
            if (rsi[i] >= 50) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend turns up
            if (rsi[i] <= 50) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals