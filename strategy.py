#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and 1w trend filter
# Donchian(20) captures breakouts in trending markets
# 1d volume > 2x 20-period average confirms breakout strength
# 1w EMA34 provides higher timeframe bias to avoid counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_Donchian20_1dVolSpike_1wEMA34"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d volume spike for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Donchian(20) on 4h
    period = 20
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(n):
        if i < period:
            upper[i] = np.nan
            lower[i] = np.nan
        else:
            upper[i] = np.max(high[i-period:i])
            lower[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band + above 1w EMA34 + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + below 1w EMA34 + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower band or trend changes
            if (close[i] < lower[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper band or trend changes
            if (close[i] > upper[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals