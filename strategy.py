#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation
# - Williams %R(14) from 6h data: long when < -80 (oversold), short when > -20 (overbought)
# - 1-week EMA(21) trend filter: only long in weekly uptrend (price > EMA21), only short in weekly downtrend (price < EMA21)
# - Volume confirmation: current 6h volume > 2.0x 20-period average to avoid false signals in low volume
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Weekly trend filter reduces counter-trend trades in strong moves, volume spike confirms momentum

name = "6h_1w_williamsr_meanrev_trend_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long when Williams %R rises above -50 (momentum fading) or weekly trend turns down
            if williams_r[i] > -50 or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R falls below -50 (momentum fading) or weekly trend turns up
            if williams_r[i] < -50 or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme + volume confirmation + weekly trend filter
            if volume_confirmed:
                # Long entry: oversold (< -80) and weekly uptrend (price > weekly EMA21)
                if williams_r[i] < -80 and close[i] > ema_21_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: overbought (> -20) and weekly downtrend (price < weekly EMA21)
                elif williams_r[i] > -20 and close[i] < ema_21_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals