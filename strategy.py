#!/usr/bin/env python3
"""
4h_ema_bounce_volume_trend_v1
Hypothesis: Price retests EMA(21) with volume confirmation and trend alignment (EMA50) provides high-probability entries in both bull and bear markets.
In bull markets, price bounces off EMA21 support; in bear markets, price rejects EMA21 resistance.
Volume filter ensures institutional participation; EMA50 confirms trend direction.
Targets 20-50 trades/year by requiring confluence of EMA21 bounce/rejection, volume spike, and EMA50 trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_bounce_volume_trend_v1"
timeframe = "4h"
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
    
    # EMA21 for dynamic support/resistance
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # EMA50 for trend filter
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA21 OR trend turns down
            if close[i] < ema21[i] or close[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above EMA21 OR trend turns up
            if close[i] > ema21[i] or close[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches/bounces off EMA21 support + volume + uptrend
            if (low[i] <= ema21[i] * 1.005 and  # Allow small tolerance for wick
                close[i] > ema21[i] and
                vol_confirm and 
                close[i] > ema50[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches/rejects EMA21 resistance + volume + downtrend
            elif (high[i] >= ema21[i] * 0.995 and  # Allow small tolerance for wick
                  close[i] < ema21[i] and 
                  vol_confirm and 
                  close[i] < ema50[i]):
                position = -1
                signals[i] = -0.25
    
    return signals