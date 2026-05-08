# This strategy is designed for daily (1D) timeframe with weekly (1W) higher timeframe trend filtering.
# It combines: 1) Weekly EMA200 trend filter, 2) Daily Donchian(20) breakout, 3) Volume confirmation (>1.5x 20-period volume EMA).
# Logic: Go long when price breaks above daily Donchian high AND price is above weekly EMA200 AND volume is strong.
# Go short when price breaks below daily Donchian low AND price is below weekly EMA200 AND volume is strong.
# Exit when price crosses back below/above the weekly EMA200.
# Designed for low-frequency, high-conviction trades to avoid overtrading and fee drag.
# Target: 20-50 trades per year (80-200 total over 4 years) to stay within profitable ranges.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1D_Donchian_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volume filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ema_20)
    
    # Calculate Donchian channels on daily timeframe (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_200_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed weekly bar for volume check
        idx_1w = 0
        while idx_1w < len(df_1w) and df_1w.iloc[idx_1w]['open_time'] <= prices.iloc[i]['open_time']:
            idx_1w += 1
        idx_1w -= 1  # last completed weekly bar
        
        if idx_1w < 0:
            vol_filter = False
        else:
            vol_1w_current = df_1w.iloc[idx_1w]['volume']
            vol_filter = vol_1w_current > 1.5 * vol_ema_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + weekly trend + volume
            long_condition = breakout_up and close[i] > ema_200_aligned[i] and vol_filter
            short_condition = breakout_down and close[i] < ema_200_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA200
            if close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA200
            if close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals