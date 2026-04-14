#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for monthly pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate monthly pivot points using prior month's OHLC
    # Approximate monthly by using 20-day period (1 trading month)
    period = 20
    if len(high_1d) < period:
        return np.zeros(n)
    
    # Rolling max/min/mean for monthly high/low/close
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    volume_series = pd.Series(volume_1d)
    
    monthly_high = high_series.rolling(window=period, min_periods=period).max().shift(1).values
    monthly_low = low_series.rolling(window=period, min_periods=period).min().shift(1).values
    monthly_close = close_series.rolling(window=period, min_periods=period).mean().shift(1).values
    monthly_volume = volume_series.rolling(window=period, min_periods=period).mean().shift(1).values
    
    # Pivot point: (H + L + C) / 3
    pp = (monthly_high + monthly_low + monthly_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - monthly_low
    s1 = 2 * pp - monthly_high
    r2 = pp + (monthly_high - monthly_low)
    s2 = pp - (monthly_high - monthly_low)
    
    # Align pivot levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: current 4h volume > 1.5x monthly average volume
    vol_series_4h = pd.Series(volume)
    avg_vol_4h = vol_series_4h.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = max(20, 20)  # 20 for volume and pivot calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(avg_vol_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 resistance with volume confirmation
            if price > r1_aligned[i] and vol > 1.5 * avg_vol_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 support with volume confirmation
            elif price < s1_aligned[i] and vol > 1.5 * avg_vol_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 support
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1 resistance
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Monthly_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0