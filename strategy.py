#!/usr/bin/env python3
"""
1h Momentum Pullback with 4h Trend Filter and Volume Confirmation
Hypothesis: In trending markets (identified by 4h EMA), pullbacks to 1h EMA provide
high-probability entries. Volume confirms institutional interest. Designed for
15-35 trades/year to avoid fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h EMA for pullback entry
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (already datetime64)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_20_1h[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session check
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1h EMA20 OR trend reverses
            if (close[i] < ema_20_1h[i] or 
                close[i] < ema_20_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 1h EMA20 OR trend reverses
            if (close[i] > ema_20_1h[i] or 
                close[i] > ema_20_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter: price vs 4h EMA20
            uptrend = close[i] > ema_20_4h_aligned[i]
            downtrend = close[i] < ema_20_4h_aligned[i]
            
            # Long: price pulls back to 1h EMA20 in uptrend + volume spike
            if (low[i] <= ema_20_1h[i] and 
                close[i] > ema_20_1h[i] and  # confirmation bounce
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: price pulls back to 1h EMA20 in downtrend + volume spike
            elif (high[i] >= ema_20_1h[i] and 
                  close[i] < ema_20_1h[i] and  # confirmation rejection
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals