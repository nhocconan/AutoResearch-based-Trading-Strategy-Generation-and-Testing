#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_1dTrend
Hypothesis: Williams Vix Fix (WVF) identifies extreme fear/greed on 6h. In bull markets (price>1d EMA50), mean revert from extreme fear (WVF>0.8). In bear markets (price<1d EMA50), mean revert from extreme greed (WVF<0.2). Uses 1d trend filter to avoid fighting the trend. Designed for 15-25 trades/year on BTC/ETH. Works in both bull (buy fear dips) and bear (sell greed spikes) by adapting mean reversion logic to regime.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Vix Fix: WVF = ((Highest High in 22-period - Low) / (Highest High in 22-period)) * 100
    # Highest High in 22-period
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    # Avoid division by zero
    wvf = np.where(highest_high != 0, ((highest_high - low) / highest_high) * 100, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for WVF calculation
    start_idx = 22
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wvf_val = wvf[i]
        ema_trend = ema_1d_aligned[i]
        
        if position == 0:
            # Flat - look for mean reentry opportunities
            # Bull regime (price > 1d EMA50): mean revert from extreme fear (WVF > 0.8)
            # Bear regime (price < 1d EMA50): mean revert from extreme greed (WVF < 0.2)
            if close_val > ema_trend:
                # Bull market: buy extreme fear
                if wvf_val > 80:  # Extreme fear
                    signals[i] = size
                    position = 1
            else:
                # Bear market: sell extreme greed
                if wvf_val < 20:  # Extreme greed (low WVF = high volatility = panic selling)
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Long - exit when fear subsides (WVF < 0.5) or at opposite extreme
            if wvf_val < 50:  # Fear subsiding
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short - exit when greed subsides (WVF > 0.5) or at opposite extreme
            if wvf_val > 50:  # Greed subsiding
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0