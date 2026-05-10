# 6H_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Build on proven 6H Camarilla R3/S3 breakout pattern by adding 1d EMA trend filter and volume spike confirmation.
# Uses prior day's Camarilla R3/S3 levels for breakout entries, confirmed by 1d EMA trend and volume > 2x average.
# Designed for 6h timeframe to capture strong trend continuation moves with low trade frequency (target: 12-37 trades/year).
# Works in both bull and bear markets by following 1d trend direction, avoiding counter-trend trades.
# Uses discrete position sizing (0.25) to minimize fee churn.

name = "6H_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    # Note: We use the same df_1d for pivot calculation
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = C + ((H-L) * 1.1 / 4)
    # S3 = C - ((H-L) * 1.1 / 4)
    camarilla_r3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    camarilla_s3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (use prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume filter: volume > 2x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + above 1d EMA + volume spike
            if (close[i] > r3_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below 1d EMA + volume spike
            elif (close[i] < s3_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or volume drops below average
            if (close[i] < s3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or volume drops below average
            if (close[i] > r3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals