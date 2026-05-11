#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses daily trend (price > EMA50) to filter Camarilla R1/S1 breakout entries on 4h.
Enters long when price breaks above R1 in daily uptrend with volume spike (>1.5x 20-period avg).
Enters short when price breaks below S1 in daily downtrend with volume spike.
Exits when price returns to Camarilla Pivot point (PP) or trend reverses.
Designed to work in both bull and bear markets by following daily EMA50 trend filter.
Targets ~25-40 trades/year via strict breakout conditions and trend filter.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price for the period
    typical_price = (high + low + close) / 3
    # Pivot point (PP)
    pp = typical_price
    # Range
    range_ = high - low
    # Camarilla levels
    r1 = pp + (range_ * 1.0833 / 2)
    s1 = pp - (range_ * 1.0833 / 2)
    r2 = pp + (range_ * 1.1666 / 2)
    s2 = pp - (range_ * 1.1666 / 2)
    r3 = pp + (range_ * 1.2500 / 2)
    s3 = pp - (range_ * 1.2500 / 2)
    return pp, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Trend Filter (EMA50) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Daily trend: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema_50, 1, -1)
    
    # Align daily EMA50 and trend to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    trend_1d_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # --- Daily Camarilla Levels (from previous day) ---
    # Use previous day's OHLC to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate Camarilla levels using previous day's data
    pp_1d, r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    # Align Camarilla levels to 4h (these represent yesterday's levels)
    pp_1d_4h = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- 4h Volume Spike Detection ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(trend_1d_4h[i]) or 
            np.isnan(pp_1d_4h[i]) or np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Daily trend direction
        daily_trend = trend_1d_4h[i]
        
        if position == 0:
            # Long: daily uptrend + price breaks above R1 + volume
            if (daily_trend == 1 and 
                close[i] > r1_1d_4h[i] and 
                close[i-1] <= r1_1d_4h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below S1 + volume
            elif (daily_trend == -1 and 
                  close[i] < s1_1d_4h[i] and 
                  close[i-1] >= s1_1d_4h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to PP or daily trend turns down
                if (close[i] <= pp_1d_4h[i] or daily_trend == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to PP or daily trend turns up
                if (close[i] >= pp_1d_4h[i] or daily_trend == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals