#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# We go long when %R crosses above -80 (oversold bounce) in uptrend, short when %R crosses below -20 (overbought rejection) in downtrend
# Confirmed by volume spike (>2x 20-period average) and filtered by 1d EMA(34) trend
# Designed for low trade frequency with mean-reversion in ranging markets and trend continuation in trending markets
# Target: 80-180 total trades over 4 years = 20-45/year

name = "4h_WilliamsR_1dTrend_Volume"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Williams %R crossovers
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    wr_cross_above_80 = (williams_r > -80) & (williams_r_prev <= -80)  # Oversold bounce
    wr_cross_below_20 = (williams_r < -20) & (williams_r_prev >= -20)  # Overbought rejection
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        wr = williams_r[i]
        wr_cross_up = wr_cross_above_80[i]
        wr_cross_down = wr_cross_below_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (oversold bounce) + uptrend + volume spike
            if (wr_cross_up and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (overbought rejection) + downtrend + volume spike
            elif (wr_cross_down and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price breaks below trend
            if (wr_cross_down or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price breaks above trend
            if (wr_cross_up or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals