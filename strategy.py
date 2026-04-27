# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h breakout at Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
# Uses 1d EMA(34) for trend direction, daily pivots for R1/S1 levels, and volume spike to avoid false breakouts.
# Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets by aligning with higher timeframe structure.
# Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 200 total.

#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla levels (based on previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance and Support levels (R1, S1)
    R1 = pivot + (range_hl * 1.1 / 6)  # R1 level
    S1 = pivot - (range_hl * 1.1 / 6)  # S1 level
    
    # Get 1d data for trend filter
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need daily pivots, 1d EMA, and volume data
    start_idx = max(2, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (volume spike)
        vol_filter = vol_current > (vol_ma_val * 2.0)
        
        if position == 0:
            # Long: price breaks above R1 with 1d uptrend and volume spike
            if close[i] > R1_aligned[i] and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with 1d downtrend and volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S1 or 1d trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R1 or 1d trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0