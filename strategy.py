#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions (below -80 for long, above -20 for short)
# 12h EMA34 provides trend filter to trade only in direction of intermediate trend
# Volume > 1.8x 20-period average confirms institutional participation
# Discrete position sizing (0.25) with mean reversion exit at Williams %R crossing -50
# Designed for ~15-30 trades/year to minimize fee drag while capturing reversals in trending markets
# Works in bull/bear via trend filter - only takes longs in uptrend, shorts in downtrend

name = "6h_WilliamsR_Extreme_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema34_12h = ema_34_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion to midpoint)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion to midpoint)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.8x 20-period average
            vol_spike = curr_volume > 1.8 * curr_vol_ma
            
            # Long when Williams %R < -80 (oversold) with 12h EMA34 uptrend and volume spike
            if curr_williams_r < -80 and curr_close > curr_ema34_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) with 12h EMA34 downtrend and volume spike
            elif curr_williams_r > -20 and curr_close < curr_ema34_12h and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals