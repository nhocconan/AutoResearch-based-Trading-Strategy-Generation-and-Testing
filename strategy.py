#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean-reversion strategy using 4h RSI(14) extreme levels with 1h Bollinger Band mean reversion and volume filter.
# Long when 4h RSI < 30 (oversold), price touches 1h lower Bollinger Band (20,2), and 1h volume > 1.2x median.
# Short when 4h RSI > 70 (overbought), price touches 1h upper Bollinger Band (20,2), and 1h volume > 1.2x median.
# Exit when price crosses 1h middle Bollinger Band (20-period SMA).
# Uses discrete position size 0.20. Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year). Uses 4h for extreme regime filter, 1h for mean-reversion timing.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data once before loop for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h Indicators: RSI(14) for extreme regime detection ===
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Get 1h data for Bollinger Bands and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # === 1h Indicators: Bollinger Bands (20,2) and Volume median ===
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    vol_1h = df_1h['volume'].values
    
    # Bollinger Bands
    sma_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Volume median for spike detection
    vol_median_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (1h)
    rsi_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1h, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1h, middle_bb)
    vol_median_aligned = align_htf_to_ltf(prices, df_1h, vol_median_20)
    vol_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_1h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 20, 20)  # RSI(14), BB(20), volume median(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(middle_bb_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_1h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        rsi = rsi_14_aligned[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        middle = middle_bb_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1h = vol_1h_aligned[i]
        
        # Price levels
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses above middle Bollinger Band (mean reversion complete)
            if price > middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses below middle Bollinger Band (mean reversion complete)
            if price < middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 1h volume > 1.2x median volume (avoid low-volume false signals)
            volume_filter = vol_1h > (vol_median * 1.2)
            
            # LONG CONDITIONS
            # 4h RSI oversold (<30) AND price touches lower Bollinger Band AND volume filter
            if rsi < 30 and low_price <= lower and volume_filter:
                signals[i] = 0.20
                position = 1
            
            # SHORT CONDITIONS
            # 4h RSI overbought (>70) AND price touches upper Bollinger Band AND volume filter
            elif rsi > 70 and high_price >= upper and volume_filter:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_RSI14_Extreme_1hBB20_2_VolumeFilter1.2x_v1"
timeframe = "1h"
leverage = 1.0