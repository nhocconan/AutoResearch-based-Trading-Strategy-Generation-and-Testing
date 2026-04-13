#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's data)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Key Camarilla levels (tighter bands for fewer trades)
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Weekly trend filter (40-period EMA for smoother trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean()
    
    # Daily volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    
    # Align all data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w.values)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.3x 20-period average daily volume
        # (conservative threshold to reduce trades)
        volume_condition = volume[i] > (volume_ma_20_1d_aligned[i] * 1.3)
        
        # Trend filter: only trade in direction of weekly trend
        long_trend = close[i] > ema_40_1w_aligned[i]
        short_trend = close[i] < ema_40_1w_aligned[i]
        
        # Entry conditions: price at key Camarilla levels with volume and trend
        # Long at S1 with volume and uptrend
        # Short at R1 with volume and downtrend
        at_support = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.003  # Within 0.3%
        at_resistance = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.003  # Within 0.3%
        
        if position == 0:
            if at_support and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif at_resistance and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or shows weakness
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or shows strength
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v4"
timeframe = "12h"
leverage = 1.0