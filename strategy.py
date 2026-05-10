# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout at Camarilla R3/S3 levels on 12-hour chart with daily trend filter and volume confirmation.
# In strong daily uptrend (price > daily EMA34), long at R3 breakout with target at S3 of next day.
# In strong daily downtrend (price < daily EMA34), short at S3 breakdown with target at R3 of next day.
# Uses volume confirmation to avoid low-liquidity whipsaws. Designed for 12h to achieve 12-37 trades/year.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d data for trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for each 1d bar (using current day's OHLC for breakout)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    rng = high_1d - low_1d
    r3_1d = close_1d + 1.1 * rng
    s3_1d = close_1d - 1.1 * rng
    
    # Volume confirmation: 24-period average volume on 1d (2 days of 12h data)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_24_1d = mean_arr(vol_1d, 24)
    
    # Align 1d trend to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align 1d Camarilla levels to 12h (using current day's levels for breakout)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Align 1d volume MA to 12h
    vol_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma_24_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        is_uptrend = close_1d[-1] > ema_34_1d[-1] if len(close_1d) > 0 else False  # Use last known daily close
        is_downtrend = close_1d[-1] < ema_34_1d[-1] if len(close_1d) > 0 else False
        
        # Volume condition: current 12h volume > 1.5x 24-period 1d average
        volume_condition = volume[i] > 1.5 * vol_ma_24_1d_aligned[i]
        
        if position == 0:
            # Breakout at R3 in uptrend: long when price breaks above R3 resistance
            if is_uptrend and close[i] > r3_1d_aligned[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Breakdown at S3 in downtrend: short when price breaks below S3 support
            elif is_downtrend and close[i] < s3_1d_aligned[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S3 of next day (target) or daily trend turns down
            # For simplicity, exit when price reaches S3 level (using same day's S3 as approximation)
            if close[i] <= s3_1d_aligned[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R3 of next day (target) or daily trend turns up
            if close[i] >= r3_1d_aligned[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals