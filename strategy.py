#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) in a 1d uptrend with volume > 1.3x average
# Short when Williams %R > -20 (overbought) in a 1d downtrend with volume > 1.3x average
# Exit when Williams %R crosses -50 (middle) or trend reverses
# Williams %R identifies mean-reversion extremes, 1d trend filters counter-trend trades,
# volume confirms momentum. Targets 20-40 trades/year for low fee drag.

name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    highest_high = rolling_max(high, 14)
    lowest_low = rolling_min(low, 14)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d[1:] - ema50_1d[:-1]  # slope: positive = uptrend
    ema50_1d_slope = np.concatenate([[0], ema50_1d_slope])  # align length
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        ema50_val = ema50_1d_aligned[i]
        ema50_slope = ema50_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), volume confirmation, 1d uptrend
            if wr < -80 and vol_conf_val and ema50_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), volume confirmation, 1d downtrend
            elif wr > -20 and vol_conf_val and ema50_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or 1d trend turns down
            if wr > -50 or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or 1d trend turns up
            if wr < -50 or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals