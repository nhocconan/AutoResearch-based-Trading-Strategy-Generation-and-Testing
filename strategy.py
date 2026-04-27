# 11:50 PM
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA trend filter and volume spike.
# Williams %R (WPR) identifies overbought/oversold conditions.
# WPR < -80 = oversold (buy signal), WPR > -20 = overbought (sell signal).
# Trend filter: 12h EMA50 - only trade in direction of higher timeframe trend.
# Volume spike confirms institutional participation.
# Designed for ~20-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    
    wpr = -100 * ((highest_high - close) / denom)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wpr[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals with trend filter
        if wpr[i] < -80:  # Oversold - potential long
            if close[i] > ema50_12h_aligned[i] and volume_filter[i]:  # Only long in uptrend
                signals[i] = 0.25
        elif wpr[i] > -20:  # Overbought - potential short
            if close[i] < ema50_12h_aligned[i] and volume_filter[i]:  # Only short in downtrend
                signals[i] = -0.25
        else:
            # Neutral zone - maintain flat
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0