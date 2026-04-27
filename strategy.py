#!/usr/bin/env python3
"""
12h_Philips_Momentum_Trend_Breakout
Hypothesis: Combines momentum acceleration (ROC) with trend (EMA) and volume confirmation to capture strong directional moves on 12h. Uses 1d for trend filter and volume spike. Designed for low trade frequency (15-25/year) to minimize fee drag while capturing major trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h ROC(12) for momentum acceleration - measures rate of change over ~6 days
    roc_period = 12
    roc = np.zeros(n)
    roc[:roc_period] = np.nan
    for i in range(roc_period, n):
        roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period] * 100
    
    # 12h EMA21 for dynamic trend
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for ROC and EMA
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(roc[i]) or np.isnan(ema21[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        mom = roc[i]
        ema_trend = ema21[i]
        ema1d_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: positive momentum acceleration + uptrend on both timeframes + volume spike
            if mom > 3.0 and ema_trend > ema1d_trend and close[i] > ema_trend and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: negative momentum acceleration + downtrend on both timeframes + volume spike
            elif mom < -3.0 and ema_trend < ema1d_trend and close[i] < ema_trend and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: momentum turns negative OR trend breaks down
            if mom < 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: momentum turns positive OR trend breaks up
            if mom > 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Philips_Momentum_Trend_Breakout"
timeframe = "12h"
leverage = 1.0