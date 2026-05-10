#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Combines 1d EMA trend filter with tighter Camarilla R3/S3 breakouts on 12h timeframe.
# Uses higher timeframe trend (1d EMA34) to filter direction, reducing false signals in choppy markets.
# Targets 12-37 trades per year on 12h timeframe with discrete position sizing (0.25) to minimize churn.
# Designed to work in both bull and bear markets by aligning with 1d trend and requiring volume confirmation.
# Focus on BTC/ETH as primary targets with stricter entry conditions to avoid overtrading.

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
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
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    camarilla_r3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    camarilla_s3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use prior 1d bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume filter: volume > 1.5x 34-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 34)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S3 (re-enters range) or volume drops below average
            if (close[i] < s3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R3 (re-enters range) or volume drops below average
            if (close[i] > r3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals