#!/usr/bin/env python3
# 6h_donchian_1w_pullback_volume_v1
# Hypothesis: 6h Donchian(20) breakout with pullback entry in direction of 1w trend, confirmed by volume spike.
# In bull markets (price > 1w EMA50): long on breakout pullback to 20-period EMA.
# In bear markets (price < 1w EMA50): short on breakdown pullback to 20-period EMA.
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_pullback_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels (20-period) on 6h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA for pullback entry
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(period20_high[i]) or 
            np.isnan(period20_low[i]) or np.isnan(ema20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime from 1w trend
        is_bull = close[i] > ema50_1w_aligned[i]  # Using current close vs aligned 1w EMA50
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low OR closes below 20 EMA
            if low[i] < period20_low[i] or close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high OR closes above 20 EMA
            if high[i] > period20_high[i] or close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # In bull market: long on breakout pullback to 20 EMA
                if is_bull and high[i] > period20_high[i] and low[i] <= ema20[i]:
                    position = 1
                    signals[i] = 0.25
                # In bear market: short on breakdown pullback to 20 EMA
                elif (not is_bull) and low[i] < period20_low[i] and high[i] >= ema20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals