#!/usr/bin/env python3
"""
12h_Trend_Reversal_With_Volume
Hypothesis: On 12h timeframe, reversal signals occur when price closes outside Bollinger Bands
with volume confirmation and daily EMA trend filter. Bollinger Bands capture volatility expansion
during reversals, while volume confirms institutional participation. Works in both bull and bear
markets by catching trend exhaustion and reversals. Target: 15-35 trades/year (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 12h close
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Volume filter: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = bb_period  # Wait for BB to be calculated
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_12h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long reversal: close below lower BB with volume in uptrend context
            if price < bb_low and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short reversal: close above upper BB with volume in downtrend context
            elif price > bb_up and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to middle BB or trend reverses
                if price > bb_middle[i] or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to middle BB or trend reverses
                if price < bb_middle[i] or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Trend_Reversal_With_Volume"
timeframe = "12h"
leverage = 1.0