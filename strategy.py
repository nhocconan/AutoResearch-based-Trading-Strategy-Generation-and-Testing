#!/usr/bin/env python3
# 1d_weekly_engulfing_volume_v1
# Hypothesis: Daily bullish/bearish engulfing patterns with weekly trend filter and volume confirmation.
# Engulfing candles signal strong momentum shifts; weekly trend ensures alignment with higher timeframe momentum.
# Volume confirms institutional participation. Works in bull markets (catching uptrends) and bear markets (catching downtrends).
# Target: 15-25 trades/year with position size 0.25 to minimize fee drag.

name = "1d_weekly_engulfing_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA trend filter (21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(vol_period, 2) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Candlestick patterns
        bullish_engulfing = (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        bearish_engulfing = (open_price[i] > close[i-1]) and (close[i] < open_price[i-1])
        
        if position == 1:  # Long position
            # Exit on bearish engulfing or trend failure
            if bearish_engulfing or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on bullish engulfing or trend failure
            if bullish_engulfing or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish engulfing with uptrend and volume
            if bullish_engulfing and close[i] > ema_1w_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish engulfing with downtrend and volume
            elif bearish_engulfing and close[i] < ema_1w_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals