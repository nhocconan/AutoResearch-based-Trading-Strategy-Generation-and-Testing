#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 3:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: H5, L3, L4, L5, H3, H4
    range_val = prev_high - prev_low
    H5 = prev_close + 1.1 * range_val / 2
    L3 = prev_close - 1.1 * range_val / 6
    L4 = prev_close - 1.1 * range_val / 4
    L5 = prev_close - 1.1 * range_val / 2
    H3 = prev_close + 1.1 * range_val / 4
    H4 = prev_close + 1.1 * range_val / 6
    
    # Align to 12h
    H5_aligned = align_htf_to_ltf(prices, df_1d, H5)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    L5_aligned = align_htf_to_ltf(prices, df_1d, L5)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Trend filter: 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 2.0 * 24-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(24, 34)  # Need enough data for volume MA and EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(H5_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(L5_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        h5 = H5_aligned[i]
        l3 = L3_aligned[i]
        l4 = L4_aligned[i]
        l5 = L5_aligned[i]
        h3 = H3_aligned[i]
        h4 = H4_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above H4 with volume and above trend
            if close[i] > h4 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below L3 with volume and below trend
            elif close[i] < l3 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below L4 (mean reversion to midpoint)
            if close[i] < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above H4 (mean reversion to midpoint)
            if close[i] > h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals