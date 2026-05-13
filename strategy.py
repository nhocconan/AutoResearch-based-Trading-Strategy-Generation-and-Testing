#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
# Long when Jaw < Teeth < Lips (bullish alignment) with 1w EMA50 uptrend and volume > 1.5x average.
# Short when Jaw > Teeth > Lips (bearish alignment) with 1w EMA50 downtrend and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams Alligator identifies trend initiation and continuation. 1w EMA50 ensures we trade with the higher timeframe trend.
# Volume confirmation confirms participation. Works in bull markets via upward alignment and in bear markets via downward alignment.

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
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
    
    # Williams Alligator (13,8,5) with SMMA (smoothed moving average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line (13-period)
    teeth = smma(close, 8)  # Red line (8-period)
    lips = smma(close, 5)   # Green line (5-period)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for Alligator (13 periods) and volume average (20 periods)
    start_idx = max(13, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish alignment (Jaw < Teeth < Lips) with 1w EMA50 uptrend and volume spike
            if (jaw[i] < teeth[i] < lips[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment (Jaw > Teeth > Lips) with 1w EMA50 downtrend and volume spike
            elif (jaw[i] > teeth[i] > lips[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment OR price below 1w EMA50 (trend change)
            if (jaw[i] > teeth[i] > lips[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment OR price above 1w EMA50 (trend change)
            if (jaw[i] < teeth[i] < lips[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals