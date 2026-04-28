#!/usr/bin/env python3
"""
1h_Volume_Weighted_CCI_Divergence
Hypothesis: Uses 1h CCI (20) for momentum and volume-weighted CCI for divergence detection. 
Trades only during 8-20 UTC session with EMA(50) trend filter on 1h to avoid counter-trend whipsaws. 
Designed for low frequency (15-35 trades/year) by requiring both CCI divergence and volume confirmation. 
Works in bull/bear by following 1h EMA50 trend direction. Targets 60-140 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price
    tp = (high + low + close) / 3.0
    
    # CCI(20) calculation
    cci_period = 20
    sma_tp = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).mean().values
    mad = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume-weighted CCI (VWCCI) for divergence
    vw_tp = tp * volume
    vw_sma_tp = pd.Series(vw_tp).rolling(window=cci_period, min_periods=cci_period).sum().values / \
                pd.Series(volume).rolling(window=cci_period, min_periods=cci_period).sum().values
    vw_mad = pd.Series(vw_tp).rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values / np.where(
        pd.Series(volume).rolling(window=cci_period, min_periods=cci_period).sum().values == 0,
        1,
        pd.Series(volume).rolling(window=cci_period, min_periods=cci_period).sum().values
    )
    vw_cci = (vw_tp - vw_sma_tp) / (0.015 * vw_mad)
    vw_cci = np.where(np.isnan(vw_cci), 0, vw_cci)
    
    # EMA(50) trend filter on 1h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.to_datetime(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and CCI to stabilize
    
    for i in range(start_idx, n):
        # Session filter
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(vw_cci[i]) or 
            np.isnan(ema_50[i]) or np.isnan(cci[i-1]) or np.isnan(vw_cci[i-1])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # CCI divergence conditions
        # Bullish divergence: price makes lower low, CCI makes higher low
        bull_div = (low[i] < low[i-1]) and (cci[i] > cci[i-1]) and (vw_cci[i] > vw_cci[i-1])
        # Bearish divergence: price makes higher high, CCI makes lower high
        bear_div = (high[i] > high[i-1]) and (cci[i] < cci[i-1]) and (vw_cci[i] < vw_cci[i-1])
        
        # Entry conditions with trend alignment
        long_entry = bull_div and uptrend and (cci[i] < -50)  # Oversold bullish divergence
        short_entry = bear_div and downtrend and (cci[i] > 50)  # Overbought bearish divergence
        
        # Exit conditions: CCI crosses zero or opposite divergence
        long_exit = (cci[i] > 0) or (high[i] > high[i-1] and cci[i] < cci[i-1])
        short_exit = (cci[i] < 0) or (low[i] < low[i-1] and cci[i] > cci[i-1])
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Volume_Weighted_CCI_Divergence"
timeframe = "1h"
leverage = 1.0