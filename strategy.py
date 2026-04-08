#!/usr/bin/env python3
"""
1h Momentum with 4h Trend and Volume Confirmation
Hypothesis: Price momentum on 1h (price > EMA20) confirmed by 4h EMA50 trend and volume spikes,
captures trending moves while avoiding false signals in ranging markets. Uses session filter (08-20 UTC)
to reduce noise. Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA20 for momentum
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_20[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_spike[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA20 or trend reverses
            if close[i] < ema_20[i] or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA20 or trend reverses
            if close[i] > ema_20[i] or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Momentum: price vs EMA20
            bullish = close[i] > ema_20[i]
            bearish = close[i] < ema_20[i]
            
            # Long: price > EMA20 + 4h uptrend + volume spike
            if (bullish and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: price < EMA20 + 4h downtrend + volume spike
            elif (bearish and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals