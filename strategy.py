#!/usr/bin/env python3
"""
1d_WilliamsVixFix_Reversal_1wTrend
Hypothesis: Williams Vix Fix identifies extreme fear/greed on 1d; reversed when 1w EMA50 trend aligns. Works in both bull/bear by fading exhaustion. Low frequency (~15 trades/year) minimizes fee drag.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Williams Vix Fix on 1d: measures fear/greed, high = fear (buy signal)
    # WVF = ((Highest Close in 22d - Low) / (Highest Close in 22d)) * 100
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low) / highest_close) * 100
    wvf_ema = pd.Series(wvf).ewm(span=9, adjust=False, min_periods=9).mean().values  # smooth
    
    # Bollinger Bands %B on 1d for mean reversion context
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pctb = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need 1w EMA50 (50*7=350 bars), WVF lookback (22), BB (20)
    start_idx = max(50*7, 22, 20) + 5  # ~355 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(wvf_ema[i]) or 
            np.isnan(bb_pctb[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        wvf_val = wvf_ema[i]
        bb_pctb_val = bb_pctb[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        
        if position == 0:
            # Look for entry: extreme fear (high WVF) in alignment with 1w trend
            # Extreme fear: WVF > 80 (90th percentile historically)
            # Mean reversion: price near lower BB
            long_condition = (wvf_val > 80 and 
                            close_val < bb_lower_val * 1.01 and  # near or below lower BB
                            close_val > ema_val)  # only long if above 1w EMA (uptrend)
            
            # Extreme greed: low WVF (< 20) indicates complacency
            # Mean reversion: price near upper BB
            short_condition = (wvf_val < 20 and 
                             close_val > bb_upper_val * 0.99 and  # near or above upper BB
                             close_val < ema_val)  # only short if below 1w EMA (downtrend)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: fear subsides (WVF drops) or trend changes
            if wvf_val < 40 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: greed subsides (WVF rises) or trend changes
            if wvf_val > 60 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsVixFix_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0