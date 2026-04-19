# 6h_ElderRay_BullPower_Trend_Align
# Hypothesis: Elder Ray indicator (Bull Power/Bear Power) combined with trend alignment from 1d EMA50.
# Bull Power measures bullish strength (high - EMA13), Bear Power measures bearish strength (low - EMA13).
# In trending markets, we take long when Bull Power > 0 and price > 1d EMA50, short when Bear Power < 0 and price < 1d EMA50.
# Uses 13-period EMA for Ray calculation and 50-period EMA for trend filter on daily timeframe.
# Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via trend filter - only trades in direction of higher timeframe trend.

name = "6h_ElderRay_BullPower_Trend_Align"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray on 6h data
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND price above 1d EMA50 (uptrend)
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND price below 1d EMA50 (downtrend)
            elif bear_power[i] < 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bull Power turns negative OR price breaks below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bear Power turns positive OR price breaks above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals