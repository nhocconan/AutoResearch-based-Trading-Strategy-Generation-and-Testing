#!/usr/bin/env python3
# 1h_volume_regime_breakout_v1
# Hypothesis: 1h strategy using 4h/1d trend filter with volume breakout entry timing.
# In bull markets: price above 4h EMA20 + 1d EMA50 + volume > 2.0x 20-period average → long
# In bear markets: price below 4h EMA20 + 1d EMA50 + volume > 2.0x 20-period average → short
# Uses volume spikes to capture momentum bursts while avoiding choppy regimes.
# Discrete sizing (±0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_regime_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d HTF data for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA20 OR volume drops below average
            if close[i] < ema_20_4h_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA20 OR volume drops below average
            if close[i] > ema_20_4h_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if in_session:
                # Volume breakout condition
                volume_breakout = volume[i] > 2.0 * volume_ma[i]
                
                if volume_breakout:
                    # Long: price above both 4h and 1d EMAs
                    if close[i] > ema_20_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                        position = 1
                        signals[i] = 0.20
                    # Short: price below both 4h and 1d EMAs
                    elif close[i] < ema_20_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals