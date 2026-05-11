#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    ema34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1d volume filter: volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Hourly Camarilla levels (based on previous day)
    # Calculate daily pivot points from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    
    # Align to hourly
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 4h trend or 1d volume data not ready
        if np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 with 4h uptrend and volume confirmation
            if (high[i] > camarilla_h3_aligned[i] and 
                close[i] > camarilla_h3_aligned[i] and
                close[i] > ema34_4h_aligned[i] and  # 4h uptrend
                volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below L3 with 4h downtrend and volume confirmation
            elif (low[i] < camarilla_l3_aligned[i] and 
                  close[i] < camarilla_l3_aligned[i] and
                  close[i] < ema34_4h_aligned[i] and  # 4h downtrend
                  volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long when price breaks below L4 or reverses against trend
            if (low[i] < camarilla_l4_aligned[i] or 
                close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short when price breaks above H4 or reverses against trend
            if (high[i] > camarilla_h4_aligned[i] or 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals