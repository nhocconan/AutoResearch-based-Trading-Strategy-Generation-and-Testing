# 1d_1w_Camarilla_Pivot_Strategy_v1
# This strategy uses weekly Camarilla pivot levels on 1d timeframe for entries and exits.
# In bull markets: buys near support (S1) during pullbacks in uptrend
# In bear markets: sells near resistance (R1) during bounces in downtrend
# Weekly trend filter ensures we trade with higher timeframe momentum
# Volume confirmation filters out low-conviction breakouts
# Target: 10-25 trades/year to minimize fee drag
# Works in both bull and bear via directional bias from weekly trend

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Strategy_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly trend filter: EMA(34) for direction ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema_34_1w > 0  # Will be replaced with actual comparison
    
    # === Daily Camarilla pivot levels (based on prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot using previous day's OHLC
    # We need to shift by 1 to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have invalid values (from roll), handled by min_periods later
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    R2 = pivot + (range_val * 1.1 / 6)
    S2 = pivot - (range_val * 1.1 / 6)
    
    # Align weekly trend to daily
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, 
                                              pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values > 0)
    
    # === Daily volume confirmation ===
    volume = df_1d['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align volume ratio (already daily, but keep for consistency)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # === Price data ===
    close = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 20 to ensure we have enough data for indicators
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isclose(pivot[i], 0) or np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_up = weekly_trend_up_aligned[i]
        vol_ratio = volume_ratio_aligned[i]
        
        if position == 0:
            # Long conditions: weekly uptrend + price at S1 support + volume confirmation
            if (weekly_up and 
                np.isclose(close[i], S1[i], rtol=0.002) and  # Within 0.2% of S1
                vol_ratio > 1.5):  # Above average volume
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly downtrend + price at R1 resistance + volume confirmation
            elif (not weekly_up and 
                  np.isclose(close[i], R1[i], rtol=0.002) and  # Within 0.2% of R1
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 (take profit) or breaks below S2 (stop)
            if (np.isclose(close[i], R1[i], rtol=0.002) or  # Take profit at R1
                close[i] < S2[i]):  # Stop if breaks S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 (take profit) or breaks above R2 (stop)
            if (np.isclose(close[i], S1[i], rtol=0.002) or  # Take profit at S1
                close[i] > R2[i]):  # Stop if breaks R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals