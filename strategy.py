#!/usr/bin/env python3
# 6H_Aroon_Cross_1dTrend_Volume
# Hypothesis: Aroon(25) crossover signals trend strength, confirmed by 1d EMA(50) direction and volume spike >2x average.
# Aroon Up > Aroon Down indicates uptrend strength; vice versa for downtrend.
# Works in bull markets via up-trend signals, in bear markets via down-trend signals.
# Uses 6h timeframe for lower frequency, targeting 50-150 total trades over 4 years.
# Discrete position sizing (0.25) minimizes fee churn.

name = "6H_Aroon_Cross_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Aroon(25) calculation
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        high_idx = np.argmax(window_high)
        low_idx = np.argmin(window_low)
        aroon_up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for Aroon and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        # Aroon signals
        aroon_up_trend = aroon_up[i] > aroon_down[i]
        aroon_down_trend = aroon_down[i] > aroon_up[i]
        
        if position == 0:
            # Long entry: Aroon up > down + price above 1d EMA + volume spike
            if (aroon_up_trend and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Aroon down > up + price below 1d EMA + volume spike
            elif (aroon_down_trend and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Aroon down > up or volume drops below average
            if (aroon_down_trend or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Aroon up > down or volume drops below average
            if (aroon_up_trend or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals