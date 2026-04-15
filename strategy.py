#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + Volume Spike + 1d Trend Filter
# Uses Williams %R (14) for overbought/oversold reversals (long when %R < -80, short when %R > -20).
# Requires volume > 1.5x 20-period median for confirmation.
# Uses 1-day EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
# Works in both bull and bear markets by aligning reversals with the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(14, n):
        # Skip if required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x median of past 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_confirm = volume[i] > 1.5 * vol_median
        
        # Long entry: Williams %R oversold (< -80) + volume confirmation + price above 1d EMA50
        if (williams_r[i] < -80 and
            vol_confirm and
            close[i] > ema_50_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + volume confirmation + price below 1d EMA50
        elif (williams_r[i] > -20 and
              vol_confirm and
              close[i] < ema_50_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or loss of trend alignment
        elif position == 1 and (williams_r[i] > -20 or close[i] < ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r[i] < -80 or close[i] > ema_50_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0