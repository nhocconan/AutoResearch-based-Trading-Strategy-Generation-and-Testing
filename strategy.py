# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Uses Williams %R (14-period) for overbought/oversold signals in the direction of 12h EMA trend.
# Volume spike confirms momentum. Designed to capture mean-reversion within trending markets.
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h trend filter: 34-period EMA on close (Williams Alligator concept)
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Williams %R (14-period) on 6h data
    wr_period = 14
    highest_high = pd.Series(high).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low).rolling(window=wr_period, min_periods=wr_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike in uptrend
            long_condition = (williams_r[i] < -80) and vol_spike[i] and uptrend
            # Short: Williams %R overbought (> -20) with volume spike in downtrend
            short_condition = (williams_r[i] > -20) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R returns above -50 or trend turns down
            if (williams_r[i] > -50) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R returns below -50 or trend turns up
            if (williams_r[i] < -50) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals