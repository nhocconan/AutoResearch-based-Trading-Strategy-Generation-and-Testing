#!/usr/bin/env python3
# 1d_donchian_breakout_1w_ema_volume_v1
# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and 1d volume confirmation.
# Weekly EMA(50) determines primary trend (only long above, short below).
# 1d Donchian(20) breakout provides entry in trend direction.
# 1d volume > 1.8x 20-period average confirms institutional participation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-30 trades/year.
# Uses weekly HTF and daily data called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Donchian channels and volume
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR weekly EMA turns bearish (price < EMA)
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR weekly EMA turns bullish (price > EMA)
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.8x 20-period average
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Long entry: price breaks above Donchian high AND above weekly EMA (uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below weekly EMA (downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals