#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above Donchian upper band + volume > 1.5x volume MA20 + price > 12h EMA34.
Short when price breaks below Donchian lower band + volume > 1.5x volume MA20 + price < 12h EMA34.
Exit when price crosses Donchian middle band (20-period SMA) or volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > (1.5 * vol_ma)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_confirmed[i]
        
        if position == 0:
            # Long: price > upper band + volume confirmed + price > 12h EMA34
            if price > upper[i] and vol_ok and price > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < lower band + volume confirmed + price < 12h EMA34
            elif price < lower[i] and vol_ok and price < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < middle band OR volume drops below average
            if price < middle[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > middle band OR volume drops below average
            if price > middle[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0