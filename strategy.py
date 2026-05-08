#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with weekly trend filter and volume confirmation
# Williams %R measures momentum: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Overbought: > -20, Oversold: < -80
# We go long when %R crosses above -80 from below (oversold bounce) in weekly uptrend
# We go short when %R crosses below -20 from above (overbought rejection) in weekly downtrend
# Confirmed by volume spike (>2x 20-period average) to avoid false signals
# Designed for low-frequency mean reversion in ranging markets and trend exhaustion in trending markets
# Target: 50-150 total trades over 4 years = 12-37/year

name = "12h_WilliamsR_1wTrend_Volume"
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
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R(14) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below + weekly uptrend + volume spike
            if i > start_idx:
                wr_prev = williams_r[i-1]
                if (wr > -80 and wr_prev <= -80 and  # crossed above -80
                    close[i] > ema50_1w_val and       # weekly uptrend
                    vol_spike):                     # volume confirmation
                    signals[i] = 0.25
                    position = 1
            # Enter short: Williams %R crosses below -20 from above + weekly downtrend + volume spike
            if i > start_idx:
                wr_prev = williams_r[i-1]
                if (wr < -20 and wr_prev >= -20 and  # crossed below -20
                    close[i] < ema50_1w_val and       # weekly downtrend
                    vol_spike):                     # volume confirmation
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or trend fails
            if (wr > -20 or close[i] < ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or trend fails
            if (wr < -80 or close[i] > ema50_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals