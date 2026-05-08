#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with weekly trend filter and volume confirmation
# We go long when price breaks above upper BB with weekly EMA(50) uptrend and volume spike.
# We go short when price breaks below lower BB with weekly EMA(50) downtrend and volume spike.
# Uses 4h timeframe targeting 20-50 trades/year. Bollinger squeeze identifies low volatility
# periods preceding breakouts. Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false breakouts.

name = "4h_Bollinger_Squeeze_Breakout_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands (20, 2.0) on 4h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Bollinger Band width for squeeze detection (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Squeeze condition: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(squeeze_condition[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_band = bb_upper[i]
        lower_band = bb_lower[i]
        is_squeeze = squeeze_condition[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + weekly uptrend + volume spike + squeeze
            if (not np.isnan(upper_band) and close[i] > upper_band and 
                close[i] > ema50_1w_val and vol_spike and is_squeeze):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + weekly downtrend + volume spike + squeeze
            elif (not np.isnan(lower_band) and close[i] < lower_band and 
                  close[i] < ema50_1w_val and vol_spike and is_squeeze):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below middle BB OR weekly trend turns down
            if (not np.isnan(bb_middle[i]) and close[i] < bb_middle[i]) or close[i] < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above middle BB OR weekly trend turns up
            if (not np.isnan(bb_middle[i]) and close[i] > bb_middle[i]) or close[i] > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals