#!/usr/bin/env python3
name = "1d_WedgeBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Wedge detection: narrowing range over 10 days
    # Upper trendline: connect recent highs
    # Lower trendline: connect recent lows
    lookback = 10
    upper_trendline = np.full(n, np.nan)
    lower_trendline = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Get recent highs and lows
        recent_highs = high[i-lookback:i]
        recent_lows = low[i-lookback:i]
        x = np.arange(lookback)
        
        # Fit linear trendlines (degree 1)
        if len(recent_highs) >= 2:
            coeffs_high = np.polyfit(x, recent_highs, 1)
            upper_trendline[i] = np.polyval(coeffs_high, lookback-1)  # Project to current point
        
        if len(recent_lows) >= 2:
            coeffs_low = np.polyfit(x, recent_lows, 1)
            lower_trendline[i] = np.polyval(coeffs_low, lookback-1)  # Project to current point
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, lookback)  # Wait for EMA and wedge formation
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_trendline[i]) or np.isnan(lower_trendline[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper trendline in uptrend + volume
            if close[i] > upper_trendline[i] and close[i] > ema_20_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower trendline in downtrend + volume
            elif close[i] < lower_trendline[i] and close[i] < ema_20_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite trendline or trend reversal
            if position == 1:
                if close[i] < lower_trendline[i] or close[i] < ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_trendline[i] or close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals