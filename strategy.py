#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams %R Pullback + 12h EMA Trend + Volume
# Hypothesis: Fade extreme Williams %R readings during pullbacks in the direction of 12h EMA trend.
# Works in bull/bear by trading with the higher timeframe trend.
# Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.

name = "4h_williamsr_pullback_12h_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Williams %R on 4h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_4h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold or trend changes
            if williams_r[i] > -20 or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought or trend changes
            if williams_r[i] < -80 or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade extreme Williams %R in direction of 12h EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend
                    if williams_r[i] < -80:  # Oversold
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if williams_r[i] > -20:  # Overbought
                        position = -1
                        signals[i] = -0.25
    
    return signals