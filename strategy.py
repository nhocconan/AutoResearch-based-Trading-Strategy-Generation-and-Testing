#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Breakout above upper band when 1w trend is up (trend following in bull/bear).
# Breakdown below lower band when 1w trend is down (trend following in bear/bull).
# Uses 1w EMA(34) for trend and 30-period volume spike for confirmation.
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1w index
    
    # Donchian(20) on daily
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 30-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1w trend to daily
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: breakout above upper band in uptrend
            if (trend_up_aligned[i] > 0.5 and  # 1w uptrend
                high[i] >= high_20[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below lower band in downtrend
            elif (trend_up_aligned[i] <= 0.5 and  # 1w downtrend
                  low[i] <= low_20[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or stop
            if trend_up_aligned[i] <= 0.5:  # 1w downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or stop
            if trend_up_aligned[i] > 0.5:  # 1w uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals