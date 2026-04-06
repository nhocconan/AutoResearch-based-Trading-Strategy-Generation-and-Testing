#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions
# In bull markets (price > 1d EMA50): buy oversold pullbacks (%R < -80)
# In bear markets (price < 1d EMA50): sell overbought rallies (%R > -20)
# Volume confirmation ensures participation
# Target: 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear by adapting to trend direction

name = "6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -80+ is oversold, -20- is overbought
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50 = ema_50.values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.3 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend: price vs 1d EMA50
        is_uptrend = close[i] > ema_50_aligned[i]
        
        if position == 1:  # long position
            # Exit: overbought OR trend reversal
            if williams_r[i] > -20 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: oversold OR trend reversal
            if williams_r[i] < -80 or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: oversold in uptrend + volume
            if is_uptrend and williams_r[i] < -80 and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought in downtrend + volume
            elif not is_uptrend and williams_r[i] > -20 and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals