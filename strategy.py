# 6h_Camarilla_R4_S4_Breakout_WeeklyTrend
# Hypothesis: Trade breakouts at Camarilla R4/S4 (strong breakout levels) on 6h with weekly trend filter.
# In bull markets, price breaks above R4 and continues up; in bear markets, breaks below S4 and continues down.
# Weekly trend filter (price vs weekly EMA50) ensures we only trade in direction of higher timeframe trend.
# Volume confirmation (volume > 1.5x 20-period average) reduces false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
# Works in bull/bear: weekly trend filter avoids counter-trend trades, R4/S4 breakouts capture strong momentum.

name = "6h_Camarilla_R4_S4_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Calculate Camarilla levels from previous 6h bar
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    # Previous period's range
    range_prev = high_shift - low_shift
    
    # Camarilla levels R4 and S4 (strong breakout levels)
    R4 = close_shift + 1.5 * range_prev / 2
    S4 = close_shift - 1.5 * range_prev / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(R4[i]) or np.isnan(S4[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with volume spike AND weekly uptrend (price > weekly EMA50)
            if close[i] > R4[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume spike AND weekly downtrend (price < weekly EMA50)
            elif close[i] < S4[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S4 OR weekly trend turns down
            if close[i] < S4[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R4 OR weekly trend turns up
            if close[i] > R4[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals