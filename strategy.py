#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # Volume spike detection (10-period average)
    vol_ma10 = np.zeros(n)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 10:
            vol_sum -= volume[i-10]
        if i < 9:
            vol_ma10[i] = 0.0
        else:
            vol_ma10[i] = vol_sum / 10.0
    
    vol_spike = volume > (vol_ma10 * 1.5)  # 50% above average
    
    # Camarilla levels from previous day
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    # Camarilla R1 and S1 levels
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 1  # ~4 hours
    
    start_idx = 1  # need previous day data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma10[i]) or 
            np.isnan(R1[i]) or 
            np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R1 AND 1d uptrend AND volume spike
            if close[i] > R1[i] and trend_up[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S1 AND 1d downtrend AND volume spike
            elif close[i] < S1[i] and trend_down[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below S1 OR trend turns down
            if close[i] < S1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above R1 OR trend turns up
            if close[i] > R1[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1-day trend filter and volume spike confirmation
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# Volume spike ensures institutional participation, reducing false breakouts
# Cooldown of 1 bar limits trades to ~20-50 per year. Position size 0.25 manages risk.
# Target: 75-200 total trades over 4 years (19-50/year)