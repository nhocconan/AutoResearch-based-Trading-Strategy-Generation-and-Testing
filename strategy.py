#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day trend filter and volume spike
# Long when Williams %R < -80 (oversold), daily EMA(34) uptrend, and volume spike
# Short when Williams %R > -20 (overbought), daily EMA(34) downtrend, and volume spike
# Williams %R identifies reversal points in ranging markets
# Daily EMA filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms momentum behind the move; avoids weak reversals
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -80 = oversold, -20 = overbought
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA to 6h timeframe (available after daily close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), daily uptrend, volume spike
            if williams_r_val < -80 and price > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), daily downtrend, volume spike
            elif williams_r_val > -20 and price < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or daily trend turns down
            if williams_r_val > -50 or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or daily trend turns up
            if williams_r_val < -50 or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals