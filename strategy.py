#!/usr/bin/env python3
# 6h_1d_ema_pullback_v2
# Strategy: 6h EMA pullback with 1d EMA trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In strong trends, price pulls back to the 21 EMA before continuing.
# The 1d EMA50 filters for higher timeframe trend direction to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (~15-30/year)
# to minimize fee drift. Works in bull markets via long pullbacks and bear markets via short pullbacks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ema_pullback_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 60 EMA for pullback zone (21-period)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 60 volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_21[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Pullback conditions: price near 21 EMA (within 0.5%)
        near_ema = abs(close[i] - ema_21[i]) / ema_21[i] < 0.005
        
        # Trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: price pulls back to 21 EMA in uptrend with volume
        if near_ema and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price pulls back to 21 EMA in downtrend with volume
        elif near_ema and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal (price crosses 50 EMA on 1d)
        elif position == 1 and not trend_bullish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not trend_bearish:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals