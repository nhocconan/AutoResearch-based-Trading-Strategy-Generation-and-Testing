#!/usr/bin/env python3
# 12h_donchian_breakout_1w_ema_volume_v1
# Hypothesis: 12h Donchian(20) breakouts confirmed by 1w EMA trend and volume spike.
# Donchian captures momentum bursts, 1w EMA filters trend direction (works in bull/bear),
# volume ensures breakout validity. Exits on opposite Donchian breakout or trend reversal.
# Target: 15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA(21)
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend turns bearish
            if close[i] < lowest_low[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend turns bullish
            if close[i] > highest_high[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long breakout: price above upper Donchian + above 1w EMA (uptrend)
                if close[i] > highest_high[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price below lower Donchian + below 1w EMA (downtrend)
                elif close[i] < lowest_low[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals