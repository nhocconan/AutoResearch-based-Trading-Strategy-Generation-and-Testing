#!/usr/bin/env python3
# 1h_4H_Camarilla_1D_Trend_Filter
# Hypothesis: Camarilla pivot breakout on 4h filtered by 1d trend (EMA50) and volume confirmation.
# Uses 1h for precise entry timing, 4h for Camarilla levels, and 1d for trend filter.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 15-37 trades/year (60-150 total over 4 years) with session filter (08-20 UTC).

name = "1h_4H_Camarilla_1D_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Camarilla Pivot Points (from previous day) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Use previous day's high, low, close for Camarilla calculation
    # Since we're on 4h timeframe, we need daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    # We'll use the most recent completed daily bar
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Camarilla levels
    range_ = prev_high - prev_low
    if range_ <= 0:
        return np.zeros(n)
    
    # Resistance levels
    r1 = prev_close + (range_ * 1.0833 / 2)
    r2 = prev_close + (range_ * 1.1666 / 2)
    r3 = prev_close + (range_ * 1.2500 / 2)
    r4 = prev_close + (range_ * 1.5000 / 2)
    
    # Support levels
    s1 = prev_close - (range_ * 1.0833 / 2)
    s2 = prev_close - (range_ * 1.1666 / 2)
    s3 = prev_close - (range_ * 1.2500 / 2)
    s4 = prev_close - (range_ * 1.5000 / 2)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation (20-period average on 1h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 with trend and volume
            if close[i] > r3 and price_above_ema and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3 with trend and volume
            elif close[i] < s3 and price_below_ema and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: price falls below R1 or trend changes
            if close[i] < r1 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price rises above S1 or trend changes
            if close[i] > s1 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals