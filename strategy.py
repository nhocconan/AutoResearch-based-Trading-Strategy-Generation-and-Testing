#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h and 1d timeframe filters to reduce noise.
# Uses 4h EMA trend direction + 1d Bollinger Bands mean reversion signals.
# Long when: 4h EMA(50) up, price touches 1d BB lower band, volume > 1.5x 20-period avg.
# Short when: 4h EMA(50) down, price touches 1d BB upper band, volume > 1.5x 20-period avg.
# Exit when price crosses 4h EMA(50) or opposite BB band touched.
# Designed for 15-25 trades/year to avoid fee drag.
# Session filter: 08-20 UTC to trade active hours only.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align BB to 1h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Calculate 20-period volume average for 1d
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # Need EMA(50) and BB(20)
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: uptrend + price at lower BB + volume spike
            if (close[i] > ema_4h_aligned[i] and  # 4h uptrend
                low[i] <= bb_lower_aligned[i] and  # touched lower BB
                volume[i] > 1.5 * vol_ma_20_aligned[i]):  # volume spike
                position = 1
                signals[i] = position_size
            # Short setup: downtrend + price at upper BB + volume spike
            elif (close[i] < ema_4h_aligned[i] and  # 4h downtrend
                  high[i] >= bb_upper_aligned[i] and  # touched upper BB
                  volume[i] > 1.5 * vol_ma_20_aligned[i]):  # volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses 4h EMA down or touches upper BB
            if (close[i] < ema_4h_aligned[i] or 
                high[i] >= bb_upper_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses 4h EMA up or touches lower BB
            if (close[i] > ema_4h_aligned[i] or 
                low[i] <= bb_lower_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_EMA_BB_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0