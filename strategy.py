#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_1dTrendFilter
Hypothesis: Williams Vix Fix (WVF) identifies extreme fear/greed on 6h; mean revert from extreme WVF readings when aligned with 1d trend (EMA34). 
In bull markets, extreme fear (low WVF) = long opportunity; in bear markets, extreme greed (high WVF) = short opportunity.
Volume confirmation filters weak signals. Targets 12-30 trades/year by requiring extreme WVF (<20 for long, >80 for short) + 1d EMA trend + volume spike.
Uses discrete sizing (0.25) to limit fee churn. Works in both bull (buy fear) and bear (sell greed) regimes.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Vix Fix: WVF = ((Highest Close in LB - Low) / (Highest Close in LB - Lowest Low in LB)) * 100
    lb = 22  # lookback period
    highest_close = pd.Series(close).rolling(window=lb, min_periods=lb).max().values
    lowest_low = pd.Series(low).rolling(window=lb, min_periods=lb).min().values
    wvf = ((highest_close - low) / (highest_close - lowest_low)) * 100
    # Handle division by zero (when highest_close == lowest_low)
    wvf = np.where((highest_close - lowest_low) == 0, 100, wvf)
    
    # Volume confirmation: volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for WVF lb (22) and 1d EMA (34)
    start_idx = max(34, lb) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(wvf[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Extreme fear (WVF < 20) = potential long opportunity in any regime
            # Extreme greed (WVF > 80) = potential short opportunity in any regime
            extreme_fear = wfv[i] < 20
            extreme_greed = wfv[i] > 80
            
            if extreme_fear and uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif extreme_greed and downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when fear subsides (WVF > 40) or trend changes
            if wvf[i] > 40 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when greed subsides (WVF < 60) or trend changes
            if wfv[i] < 60 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_1dTrendFilter"
timeframe = "6h"
leverage = 1.0