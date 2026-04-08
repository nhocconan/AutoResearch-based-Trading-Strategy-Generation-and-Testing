#!/usr/bin/env python3
# 12h_1d_volatility_breakout_v1
# Hypothesis: 12-hour volatility breakout with volume confirmation and 1-day trend filter.
# Long: price breaks above Donchian(20) high AND volume > 1.5x 20-period average volume AND 1-day close > 1-day SMA(50).
# Short: price breaks below Donchian(20) low AND volume > 1.5x 20-period average volume AND 1-day close < 1-day SMA(50).
# Exit: price crosses the 12-period Donchian midpoint or momentum reverses.
# Designed to capture volatility expansion moves in both trending and ranging markets with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-period Donchian channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(12, n):
        donch_high[i] = np.max(high[i-12:i])
        donch_low[i] = np.min(low[i-12:i])
    donch_mid = (donch_high + donch_low) / 2
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day SMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        sma_50 = sma_50_1d_aligned[i]
        
        if np.isnan(avg_vol) or np.isnan(upper) or np.isnan(lower) or np.isnan(sma_50):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long position
            if price < mid:  # Exit when price crosses below midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > mid:  # Exit when price crosses above midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > upper and vol_surge and sma_50 > 0:  # Long breakout with volume and uptrend
                position = 1
                signals[i] = 0.25
            elif price < lower and vol_surge and sma_50 > 0:  # Short breakout with volume and uptrend (for mean reversion in ranging)
                position = -1
                signals[i] = -0.25
    
    return signals