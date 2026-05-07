# 1d_HighLowBreakout_WeeklyTrend_Volume
# Hypothesis: On daily chart, buy when price breaks above daily high of prior 3 days with volume confirmation and weekly uptrend,
# sell when price breaks below daily low of prior 3 days with volume confirmation and weekly downtrend.
# Uses weekly trend filter to capture multi-day momentum while avoiding false breakouts in ranging markets.
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull and bear markets.
timeframe = "1d"
name = "1d_HighLowBreakout_WeeklyTrend_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA trend filter (8-period)
    ema_1w = pd.Series(df_1w['close']).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume spike: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Calculate 3-day high and low
            high_3d = np.max(high[i-3:i]) if i >= 3 else np.nan
            low_3d = np.min(low[i-3:i]) if i >= 3 else np.nan
            
            if np.isnan(high_3d) or np.isnan(low_3d):
                continue
            
            # Long: price breaks above 3-day high + weekly uptrend + volume spike
            if close[i] > high_3d and ema_1w_aligned[i] > ema_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 3-day low + weekly downtrend + volume spike
            elif close[i] < low_3d and ema_1w_aligned[i] < ema_1w_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below 3-day low or weekly trend turns down
            low_3d = np.min(low[i-3:i]) if i >= 3 else np.nan
            if not np.isnan(low_3d) and close[i] < low_3d:
                signals[i] = 0.0
                position = 0
            elif ema_1w_aligned[i] < ema_1w_aligned[i-1]:  # Weekly trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above 3-day high or weekly trend turns up
            high_3d = np.max(high[i-3:i]) if i >= 3 else np.nan
            if not np.isnan(high_3d) and close[i] > high_3d:
                signals[i] = 0.0
                position = 0
            elif ema_1w_aligned[i] > ema_1w_aligned[i-1]:  # Weekly trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals