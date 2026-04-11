#!/usr/bin/env python3
# 1h_4h_1d_momentum_volume_v1
# Strategy: 1h momentum with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Momentum breakouts aligned with higher timeframe trends and volume capture
# sustained moves while avoiding false breakouts. Works in both bull (trend following) and
# bear (mean reversion during oversold bounces) by using momentum direction + volume.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_momentum_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h momentum (ROC 6-period)
    roc_period = 6
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period]
    
    # 1h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(roc[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend alignment: both 4h and 1d EMAs agree
        uptrend_4h = close[i] > ema_20_4h_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_4h = close[i] < ema_20_4h_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        # Momentum condition
        mom_up = roc[i] > 0.005  # 0.5% momentum
        mom_down = roc[i] < -0.005  # -0.5% momentum
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        # Long: Uptrend on both timeframes + positive momentum + volume
        if uptrend_4h and uptrend_1d and mom_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Downtrend on both timeframes + negative momentum + volume
        elif downtrend_4h and downtrend_1d and mom_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Momentum divergence or trend change
        elif position == 1 and (mom_down or not uptrend_4h or not uptrend_1d):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (mom_up or not downtrend_4h or not downtrend_1d):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals