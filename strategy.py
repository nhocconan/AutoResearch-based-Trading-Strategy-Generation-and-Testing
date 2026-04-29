#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 12h EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; mean reversion trades fade extremes
# 12h EMA50 filter ensures trades align with intermediate trend to avoid counter-trend whipsaws
# Volume spike (>2.0x 20-period average) confirms institutional participation at turning points
# Discrete sizing (0.25) minimizes fee churn; works in both bull/bear via trend filter adaptation
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_WilliamsR_MeanRev_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 20-period average volume for confirmation (on 6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # 12h EMA50, Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R rises above -50 (overbought) OR volume spike fades
            if curr_wr > -50 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (oversold) OR volume spike fades
            if curr_wr < -50 or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + price above 12h EMA50 + volume confirmation
            if (curr_wr < -80 and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) + price below 12h EMA50 + volume confirmation
            elif (curr_wr > -20 and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals