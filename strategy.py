#!/usr/bin/env python3
# 6h_MarketStructure_Breakout_VolumeFilter
# Hypothesis: Combine market structure (higher highs/lows) with 12h EMA trend filter and volume confirmation to capture breakouts in both bull and bear markets.
# Market structure ensures trading with the trend, EMA filter avoids counter-trend trades, volume confirms institutional participation.
# Works in bull markets by buying higher low breakouts; in bear markets by selling lower high breakdowns.
# 6h timeframe balances trade frequency and signal quality.

name = "6h_MarketStructure_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 trend filter on 12h timeframe
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate market structure: higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend)
    # Look back 20 periods for swing points
    lookback = 20
    hh = np.full(n, np.nan)  # higher high
    hl = np.full(n, np.nan)  # higher low
    lh = np.full(n, np.nan)  # lower high
    ll = np.full(n, np.nan)  # lower low
    
    for i in range(lookback, n):
        # Higher High: current high > highest high in lookback period
        hh[i] = high[i] > np.max(high[i-lookback:i])
        # Higher Low: current low > lowest low in lookback period
        hl[i] = low[i] > np.min(low[i-lookback:i])
        # Lower High: current high < highest high in lookback period
        lh[i] = high[i] < np.max(high[i-lookback:i])
        # Lower Low: current low < lowest low in lookback period
        ll[i] = low[i] < np.min(low[i-lookback:i])
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(hh[i]) or np.isnan(hl[i]) or
            np.isnan(lh[i]) or np.isnan(ll[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: higher high + higher low (uptrend structure) + above 12h EMA20 + volume confirmation
            if hh[i] and hl[i] and close[i] > ema20_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: lower high + lower low (downtrend structure) + below 12h EMA20 + volume confirmation
            elif lh[i] and ll[i] and close[i] < ema20_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if market structure breaks (lower low) or price falls below 12h EMA20
            if ll[i] or close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if market structure breaks (higher high) or price rises above 12h EMA20
            if hh[i] or close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals