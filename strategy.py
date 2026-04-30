#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on 12h timeframe.
# Entry when %R crosses above/below -50 with 1d EMA34 trend alignment and volume spike.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
# Works in bull markets (breakouts continue trend) and bear markets (mean reversion from extremes).

name = "12h_WilliamsR_1dEMA34_Volume_v1"
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
    
    # Calculate 1d EMA34 for HTF trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Williams %R ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average ON LTF
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Williams %R crossing -50 and 1d trend filter
            if curr_volume_spike:
                # Bullish: Williams %R crosses above -50 (oversold to neutral) + price above 1d EMA34
                if curr_williams_r > -50 and close[i] > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -50 (overbought to neutral) + price below 1d EMA34
                elif curr_williams_r < -50 and close[i] < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) or loses 1d trend
            if curr_williams_r > -20 or close[i] < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) or loses 1d trend
            if curr_williams_r < -80 or close[i] > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals