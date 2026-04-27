#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# Strategy: In uptrend (price > 4h EMA50), buy when Williams %R < -80 (oversold pullback)
# In downtrend (price < 4h EMA50), sell when Williams %R > -20 (overbought rally)
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    willr = -100 * ((highest_high - close) / hl_range)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 50-period EMA on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Uptrend: price above 4h EMA50
        if close[i] > ema50_4h_aligned[i]:
            # Buy on oversold pullback (Williams %R < -80) with volume confirmation
            if willr[i] < -80 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Exit long when overbought (Williams %R > -20) or trend changes
            elif willr[i] > -20 or close[i] <= ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold long position
                signals[i] = 0.20
        
        # Downtrend: price below 4h EMA50
        elif close[i] < ema50_4h_aligned[i]:
            # Sell on overbought rally (Williams %R > -20) with volume confirmation
            if willr[i] > -20 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            # Exit short when oversold (Williams %R < -80) or trend changes
            elif willr[i] < -80 or close[i] >= ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold short position
                signals[i] = -0.20
        
        # No trend (price == EMA) - stay flat
        else:
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1h_WilliamsR_4hEMA50_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0