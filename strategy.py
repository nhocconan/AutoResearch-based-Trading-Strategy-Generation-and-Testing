#!/usr/bin/env python3
# 6h_ema_crossover_volatility_filter_v1
# Hypothesis: 6h EMA crossover with volatility regime filter (ATR-based chop) to avoid whipsaws.
# Long when fast EMA crosses above slow EMA AND volatility is low (choppy market = mean reversion fading).
# Short when fast EMA crosses below slow EMA AND volatility is low.
# Uses daily trend filter (price > daily EMA200 for longs, < for shorts) to align with higher timeframe.
# Volatility regime: ATR(14) / ATR(50) < 0.8 indicates low volatility/chop.
# Designed to work in both bull and bear markets by using volatility filter to avoid false breakouts.
# Target: 12-25 trades/year (50-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_crossover_volatility_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA crossover system (6h timeframe)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_slow = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volatility regime filter: ATR ratio to detect choppy markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / atr50  # Low ratio = chop/consolidation
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in low volatility/choppy markets (mean reversion regime)
        low_volatility = atr_ratio[i] < 0.8
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR volatility expands (breakout)
            if (ema_fast[i] < ema_slow[i]) or (atr_ratio[i] > 1.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR volatility expands (breakout)
            if (ema_fast[i] > ema_slow[i]) or (atr_ratio[i] > 1.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for EMA crossover with volatility and trend filters
            bullish_cross = (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1])
            bearish_cross = (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1])
            
            bullish_setup = bullish_cross and low_volatility and (close[i] > ema200_1d_aligned[i])
            bearish_setup = bearish_cross and low_volatility and (close[i] < ema200_1d_aligned[i])
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals