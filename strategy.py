#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Trade long when price breaks above 20-day high with 1w EMA50 uptrend and volume spike
# Trade short when price breaks below 20-day low with 1w EMA50 downtrend and volume spike
# Uses discrete position sizing (0.25) to limit churn and drawdown in ranging markets

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w volume spike (2x 20-period average)
    vol_ma_1w = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1w = df_1w['volume'].values > (vol_ma_1w * 2.0)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1d Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to be valid
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-day high AND 1w EMA50 uptrend AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema50_1w_aligned[i] and vol_spike_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low AND 1w EMA50 downtrend AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_1w_aligned[i] and vol_spike_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day low OR trend fails
            if (close[i] < lowest_low[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day high OR trend fails
            if (close[i] > highest_high[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals