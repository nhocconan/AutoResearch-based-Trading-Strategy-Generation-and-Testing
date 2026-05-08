#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams %R with 1-day EMA34 filter and volume spike confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND 1-day EMA34 rising AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 (overbought rejection) AND 1-day EMA34 falling AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back through -50 (mean reversion) or opposite signal occurs.
# Williams %R identifies momentum extremes; EMA34 filters for higher timeframe trend; volume confirms participation.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk via mean-reversion exits.

name = "12h_WilliamsR_1dEMA34_Volume"
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
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20, EMA34 rising, volume filter
            long_cond = (williams_r[i] > -20) and (williams_r[i-1] <= -20) and ema34_rising[i] and volume_filter[i]
            # Short conditions: Williams %R crosses below -80, EMA34 falling, volume filter
            short_cond = (williams_r[i] < -80) and (williams_r[i-1] >= -80) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (mean reversion) or opposite signal
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (mean reversion) or opposite signal
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals