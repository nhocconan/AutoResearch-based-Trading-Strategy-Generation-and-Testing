#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_camarilla_pivot_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA for trend direction (20-period)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d Camarilla pivot levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to 1h timeframe (shifted by 1 day)
    H4_1h = align_htf_to_ltf(prices, df_1d, H4)
    H3_1h = align_htf_to_ltf(prices, df_1d, H3)
    H2_1h = align_htf_to_ltf(prices, df_1d, H2)
    H1_1h = align_htf_to_ltf(prices, df_1d, H1)
    L1_1h = align_htf_to_ltf(prices, df_1d, L1)
    L2_1h = align_htf_to_ltf(prices, df_1d, L2)
    L3_1h = align_htf_to_ltf(prices, df_1d, L3)
    L4_1h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if not in session or any data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
        if (np.isnan(H4_1h[i]) or np.isnan(H3_1h[i]) or np.isnan(H2_1h[i]) or 
            np.isnan(H1_1h[i]) or np.isnan(L1_1h[i]) or np.isnan(L2_1h[i]) or 
            np.isnan(L3_1h[i]) or np.isnan(L4_1h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L2 (strong support broken) OR trend turns bearish
            if close[i] < L2_1h[i] or ema_4h_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above H2 (strong resistance broken) OR trend turns bullish
            if close[i] > H2_1h[i] or ema_4h_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume and session must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above H3 with volume in bullish 4h trend
            # OR price bounces from L3/L4 with volume in bullish 4h trend
            if ema_4h_aligned[i] < close[i]:  # 4h bullish trend
                if ((close[i] > H3_1h[i] and close[i-1] <= H3_1h[i-1]) or  # Breakout above H3
                    ((close[i] > L3_1h[i] and close[i-1] <= L3_1h[i-1]) or  # Bounce from L3
                     (close[i] > L4_1h[i] and close[i-1] <= L4_1h[i-1]))):  # Bounce from L4
                    position = 1
                    signals[i] = 0.20
            # Short entry: price breaks below L3 with volume in bearish 4h trend
            # OR price rejects from H3/H4 with volume in bearish 4h trend
            elif ema_4h_aligned[i] > close[i]:  # 4h bearish trend
                if ((close[i] < L3_1h[i] and close[i-1] >= L3_1h[i-1]) or  # Breakdown below L3
                      ((close[i] < H3_1h[i] and close[i-1] >= H3_1h[i-1]) or  # Rejection from H3
                       (close[i] < H4_1h[i] and close[i-1] >= H4_1h[i-1]))):  # Rejection from H4
                    position = -1
                    signals[i] = -0.20
    
    return signals