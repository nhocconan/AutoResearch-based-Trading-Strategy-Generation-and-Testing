#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
# Uses Alligator jaws (13-period smoothed median) and teeth (8-period smoothed median).
# Long when lips (3-period) cross above jaws in uptrend, short when lips cross below jaws in downtrend.
# 1w EMA50 trend filter ensures alignment with higher timeframe trend.
# Volume spike confirms momentum. Target: 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components on 12h data
    median_price = (high + low) / 2
    
    # Smoothed median prices (SMMA-like using Wilder's smoothing)
    def smoothed_ma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components: lips (3), teeth (8), jaws (13)
    lips = smoothed_ma(median_price, 3)
    teeth = smoothed_ma(median_price, 8)
    jaws = smoothed_ma(median_price, 13)
    
    # Align Alligator components to 12h timeframe (already on 12h, no alignment needed)
    # But we still need to ensure proper handling of NaN values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (00-23 UTC - full day for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after Alligator jaws period
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or np.isnan(jaws[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # For 12h timeframe, we can trade anytime (no session filter needed)
        
        if position == 0:
            # Long: Lips cross above jaws in uptrend with volume confirmation
            if (lips[i] > jaws[i] and lips[i-1] <= jaws[i-1] and  # crossover
                close[i] > ema_50_1w_aligned[i] and  # 1w uptrend
                volume[i] > 1.5 * vol_avg_20[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below jaws in downtrend with volume confirmation
            elif (lips[i] < jaws[i] and lips[i-1] >= jaws[i-1] and  # crossover
                  close[i] < ema_50_1w_aligned[i] and  # 1w downtrend
                  volume[i] > 1.5 * vol_avg_20[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Lips cross back through teeth (weaker signal) or opposite crossover
            if position == 1:
                if lips[i] < teeth[i]:  # lips cross below teeth - exit long
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i]:  # lips cross above teeth - exit short
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0