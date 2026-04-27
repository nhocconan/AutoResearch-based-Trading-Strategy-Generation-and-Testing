# 6h_WeeklyPivotBreakout_12hTrend_VolumeFilter
# Hypothesis: Breakouts from weekly pivot levels (S4/S5 for long, R4/R5 for short) with 12h trend filter and volume confirmation.
# Weekly pivot provides key institutional levels, 12h trend filters for direction, volume confirms breakout strength.
# Designed to capture strong moves while filtering false breakouts. Works in both bull/bear by following 12h trend.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    open_w = df_w['open'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_w + low_w + close_w) / 3.0
    
    # Support levels
    S1 = 2 * pivot - high_w
    S2 = pivot - (high_w - low_w)
    S3 = low_w + 2 * (pivot - high_w)
    S4 = S3 - (high_w - low_w)
    S5 = S4 - (high_w - low_w)
    
    # Resistance levels
    R1 = 2 * pivot - low_w
    R2 = pivot + (high_w - low_w)
    R3 = high_w + 2 * (pivot - low_w)
    R4 = R3 + (high_w - low_w)
    R5 = R4 + (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    S4_aligned = align_htf_to_ltf(prices, df_w, S4)
    S5_aligned = align_htf_to_ltf(prices, df_w, S5)
    R4_aligned = align_htf_to_ltf(prices, df_w, R4)
    R5_aligned = align_htf_to_ltf(prices, df_w, R5)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need weekly pivots, 12h EMA, and daily volume MA
    start_idx = max(2, 20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S4_aligned[i]) or np.isnan(S5_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(R5_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_12h = ema_20_12h_aligned[i]
        
        # Current weekly pivot levels
        S4_now = S4_aligned[i]
        S5_now = S5_aligned[i]
        R4_now = R4_aligned[i]
        R5_now = R5_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Weekly pivot breakout with volume and 12h trend alignment
        if position == 0:
            # Long: price breaks above R4 with volume + 12h uptrend
            if price_now > R4_now and vol_filter and price_now > trend_12h:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 with volume + 12h downtrend
            elif price_now < S4_now and vol_filter and price_now < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below S5 or 12h trend turns down
            if price_now < S5_now or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R5 or 12h trend turns up
            if price_now > R5_now or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivotBreakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0