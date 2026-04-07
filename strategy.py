#!/usr/bin/env python3
"""
1d_volume_breakout_1w_trend_v1
Hypothesis: On daily timeframe, breakouts above 20-day high with volume surge and weekly trend filter capture strong moves. Works in bull (continuation of uptrends) and bear (shorting breakdowns in downtrends) by using weekly trend direction. Targets 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volume_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    weekly_close_series = pd.Series(w_close)
    ema20_w = weekly_close_series.ewm(span=20, adjust=False).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # Daily indicators
    # 20-day high for breakout
    high_series = pd.Series(high)
    high_20 = high_series.rolling(window=20, min_periods=20).max()
    high_20 = high_20.values
    
    # 20-day low for breakdown
    low_series = pd.Series(low)
    low_20 = low_series.rolling(window=20, min_periods=20).min()
    low_20 = low_20.values
    
    # Volume ratio: current volume / 20-day average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after 20-day window
        # Skip if weekly EMA not available
        if np.isnan(ema20_w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = ema20_w_aligned[i] > ema20_w_aligned[i-1] if i > 0 else False
        weekly_downtrend = ema20_w_aligned[i] < ema20_w_aligned[i-1] if i > 0 else False
        
        # Volume confirmation (at least 1.5x average)
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit when price drops below 20-day low OR weekly trend turns down
            if close[i] < low_20[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price rises above 20-day high OR weekly trend turns up
            if close[i] > high_20[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above 20-day high with weekly uptrend and volume
            if high[i] > high_20[i] and weekly_uptrend and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: breakdown below 20-day low with weekly downtrend and volume
            elif low[i] < low_20[i] and weekly_downtrend and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals