#!/usr/bin/env python3
"""
12H_Vortex_RSI_1wTrend_Filter
Hypothesis: Combines Vortex indicator (VI+ > VI- for uptrend) with RSI mean-reversion on 12h timeframe, filtered by weekly trend. In bull markets, VI+ dominance with RSI < 40 signals long; in bear markets, VI- dominance with RSI > 60 signals short. Weekly trend filter ensures alignment with higher timeframe momentum. Designed for low trade frequency (12-37/year) to minimize fee drag, using discrete position sizing (0.25) to reduce churn.
"""

name = "12H_Vortex_RSI_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly trend filter: EMA 34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Vortex indicator (14-period) on 12h
    period = 14
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    vm_plus = np.abs(high - low[:-1])
    vm_minus = np.abs(low - high[:-1])
    
    vi_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = vi_plus / tr_sum
    vi_minus = vi_minus / tr_sum
    
    # RSI (14-period) on 12h
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(rsi[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: VI+ > VI- (bullish vortex) + RSI < 40 (oversold) + weekly uptrend + volume confirmation
            if (vi_plus[i] > vi_minus[i] and 
                rsi[i] < 40 and 
                is_uptrend and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: VI- > VI+ (bearish vortex) + RSI > 60 (overbought) + weekly downtrend + volume confirmation
            elif (vi_minus[i] > vi_plus[i] and 
                  rsi[i] > 60 and 
                  is_downtrend and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- > VI+ (vortex bearish crossover) or RSI > 70 (overbought)
            if vi_minus[i] > vi_plus[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VI+ > VI- (vortex bullish crossover) or RSI < 30 (oversold)
            if vi_plus[i] > vi_minus[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals