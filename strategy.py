#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_Breakout_1dTrend_Volume"
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
    
    # Get daily data once for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian breakout (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Donchian and daily indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x daily average volume
        vol_filter = volume[i] > (vol_avg_aligned[i] * 1.5)
        
        if position == 0:
            # Long breakout: price breaks above Donchian high + trend filter + volume
            long_cond = (close[i] > donchian_high[i]) and (close[i] > ema50_aligned[i]) and vol_filter
            
            # Short breakdown: price breaks below Donchian low + trend filter + volume
            short_cond = (close[i] < donchian_low[i]) and (close[i] < ema50_aligned[i]) and vol_filter
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or trend reverses
            if (close[i] < donchian_low[i]) or (close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or trend reverses
            if (close[i] > donchian_high[i]) or (close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian breakout captures medium-term trends, filtered by daily EMA50 trend and volume spike.
# Long when price breaks above 20-period Donchian high, above daily EMA50 (uptrend), and with volume confirmation.
# Short when price breaks below 20-period Donchian low, below daily EMA50 (downtrend), and with volume confirmation.
# Exits when price reverses back into the Donchian channel or trend changes.
# Works in bull markets by catching breakouts and in bear markets by catching breakdowns.
# Volume filter ensures breakouts have institutional participation, reducing false signals.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.