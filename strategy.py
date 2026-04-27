# 4H_MultiTimeframe_Pivots_Trend_Strategy
# Combines daily Camarilla pivots with 4h trend filters and volume confirmation
# Designed to capture momentum moves in both bull and bear markets
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
# Uses higher timeframe structure to filter lower timeframe entries
# Features: Camarilla pivot levels, EMA trend filter, volume spike confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = volumes = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        # Fallback to daily if weekly data insufficient
        df_1w = df_1d.copy()
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla formulas: 
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.1*(High-Low)
    # H2 = Close + 0.55*(High-Low)
    # H1 = Close + 0.275*(High-Low)
    # L1 = Close - 0.275*(High-Low)
    # L2 = Close - 0.55*(High-Low)
    # L3 = Close - 1.1*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    camarilla_H4 = np.full(len(close_1d), np.nan)
    camarilla_H3 = np.full(len(close_1d), np.nan)
    camarilla_H2 = np.full(len(close_1d), np.nan)
    camarilla_H1 = np.full(len(close_1d), np.nan)
    camarilla_L1 = np.full(len(close_1d), np.nan)
    camarilla_L2 = np.full(len(close_1d), np.nan)
    camarilla_L3 = np.full(len(close_1d), np.nan)
    camarilla_L4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_H4[i] = prev_close + 1.5 * range_val
        camarilla_H3[i] = prev_close + 1.1 * range_val
        camarilla_H2[i] = prev_close + 0.55 * range_val
        camarilla_H1[i] = prev_close + 0.275 * range_val
        camarilla_L1[i] = prev_close - 0.275 * range_val
        camarilla_L2[i] = prev_close - 0.55 * range_val
        camarilla_L3[i] = prev_close - 1.1 * range_val
        camarilla_L4[i] = prev_close - 1.5 * range_val
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_4h = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_4h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_H2_4h = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_H1_4h = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_4h = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    camarilla_L2_4h = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_L3_4h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_4h = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Calculate 4h EMA trend filter (20-period)
    ema_period = 20
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate weekly EMA trend filter (50-period) from weekly data
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        # Fallback to daily EMA
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Calculate volume spike detector (20-period average)
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(ema_period, vol_period, 2)  # Need EMA and volume data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_H1_4h[i]) or np.isnan(camarilla_L1_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filters
        ema_bullish = price > ema_20[i]
        ema_bearish = price < ema_20[i]
        
        # Weekly trend filter (if available)
        weekly_bullish = True
        weekly_bearish = True
        if not np.isnan(ema_50_1w_aligned[i]):
            weekly_bullish = price > ema_50_1w_aligned[i]
            weekly_bearish = price < ema_50_1w_aligned[i]
        
        # Volume confirmation: require at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price crosses above Camarilla H1 with bullish trend and volume
            if (price > camarilla_H1_4h[i] and ema_bullish and weekly_bullish and 
                volume_confirmation):
                signals[i] = size
                position = 1
            # Short entry: price crosses below Camarilla L1 with bearish trend and volume
            elif (price < camarilla_L1_4h[i] and ema_bearish and weekly_bearish and 
                  volume_confirmation):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below Camarilla L1 or trend turns bearish
            if (price < camarilla_L1_4h[i] or not ema_bullish or not weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above Camarilla H1 or trend turns bullish
            if (price > camarilla_H1_4h[i] or not ema_bearish or not weekly_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_MultiTimeframe_Pivots_Trend_Strategy"
timeframe = "4h"
leverage = 1.0