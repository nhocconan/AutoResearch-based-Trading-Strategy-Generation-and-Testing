#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h/1d for signal direction (trend + volatility regime), 1h only for entry timing precision
# Designed for 15-35 trades/year on 1h to minimize fee drag while capturing institutional breakouts
# Works in bull markets via longs in uptrend, bear markets via shorts in downtrend, range via fade at extremes

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volatility regime (Choppiness Index proxy via ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d range (high - low) for choppiness proxy
    daily_range = high_1d - low_1d
    # Choppiness Index approximation: current ATR relative to average daily range
    # Low CHOP = trending (use breakout), High CHOP = ranging (fade extremes)
    avg_daily_range = pd.Series(daily_range).ewm(span=10, adjust=False, min_periods=10).mean().values
    chop_ratio = atr_14_1d / avg_daily_range  # Higher = more choppy
    chop_ratio_aligned = align_htf_to_ltf(prices, df_1d, chop_ratio)
    
    # Calculate 1h Camarilla pivot points (based on previous day)
    # We need previous day's OHLC - resample 1h to get daily, but use HTF helper properly
    # Instead, calculate camarilla from 1d data and align
    camarilla_period = 20  # lookback for pivot calculation
    
    # Get rolling 1d OHLC for camarilla calculation
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < camarilla_period:
        return np.zeros(n)
    
    high_1d_roll = df_1d_for_camarilla['high'].rolling(window=camarilla_period, min_periods=camarilla_period).max().values
    low_1d_roll = df_1d_for_camarilla['low'].rolling(window=camarilla_period, min_periods=camarilla_period).min().values
    close_1d_roll = df_1d_for_camarilla['close'].rolling(window=camarilla_period, min_periods=camarilla_period).last().values
    
    # Calculate camarilla levels from previous day's range
    rally = high_1d_roll - low_1d_roll
    camarilla_h3 = close_1d_roll + rally * 1.1 / 4  # R3
    camarilla_l3 = close_1d_roll - rally * 1.1 / 4  # S3
    camarilla_h4 = close_1d_roll + rally * 1.1 / 2  # R4
    camarilla_l4 = close_1d_roll - rally * 1.1 / 2  # S4
    
    # Align camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_l4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade during session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade breakouts in low chop (trending), fade in high chop (ranging)
        is_trending = chop_ratio_aligned[i] < 0.6  # Low chop = trending
        is_ranging = chop_ratio_aligned[i] > 0.8   # High chop = ranging
        
        if position == 0:
            if is_trending:
                # Breakout long: price breaks above H3 with 4h uptrend and volume spike
                if (close[i] > h3_aligned[i] and 
                    close[i] > ema_50_4h_aligned[i] and  # 4h uptrend
                    volume_spike[i]):
                    signals[i] = 0.20
                    position = 1
                # Breakout short: price breaks below L3 with 4h downtrend and volume spike
                elif (close[i] < l3_aligned[i] and 
                      close[i] < ema_50_4h_aligned[i] and  # 4h downtrend
                      volume_spike[i]):
                    signals[i] = -0.20
                    position = -1
            elif is_ranging:
                # Fade long: price touches L4 with mean reversion expectation
                if (close[i] <= l4_aligned[i] and 
                    close[i] > ema_50_4h_aligned[i]):  # Still in broader uptrend
                    signals[i] = 0.20
                    position = 1
                # Fade short: price touches H4 with mean reversion expectation
                elif (close[i] >= h4_aligned[i] and 
                      close[i] < ema_50_4h_aligned[i]):  # Still in broader downtrend
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price reaches opposite camarilla level or trend fails
            if (close[i] >= h3_aligned[i] or  # Take profit at H3
                close[i] < ema_50_4h_aligned[i]):  # Stop if trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reaches opposite camarilla level or trend fails
            if (close[i] <= l3_aligned[i] or  # Take profit at L3
                close[i] > ema_50_4h_aligned[i]):  # Stop if trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals