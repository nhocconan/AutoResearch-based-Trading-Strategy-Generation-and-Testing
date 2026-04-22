#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + regime filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Trend filter: EMA(34) slope (rising/falling)
# Entry: Long when Bull Power > 0 AND EMA(34) rising, Short when Bear Power > 0 AND EMA(34) falling
# Exit when signal reverses or power crosses zero
# Works in bull/bear by following EMA trend direction with Elder Ray momentum confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter and slope
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = np.diff(ema_34_1d, prepend=ema_34_1d[0])  # positive = rising
    
    # Align EMA and slope to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Elder Ray components on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = ema_13 - low   # Bear Power: EMA(13) - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND EMA(34) rising
            if bull_power[i] > 0 and ema_34_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive AND EMA(34) falling
            elif bear_power[i] > 0 and ema_34_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Bull Power turns negative OR EMA(34) turns falling
                if bull_power[i] <= 0 or ema_34_slope_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bear Power turns negative OR EMA(34) turns rising
                if bear_power[i] <= 0 or ema_34_slope_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA34Trend"
timeframe = "6h"
leverage = 1.0