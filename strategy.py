#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above R1 AND close > 12h EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below S1 AND close < 12h EMA50 AND volume > 1.5x 20-bar avg
# Exit when price reverts to Camarilla midpoint (R1+S1)/2 or volume drops
# Target: 25-40 trades/year via tight entry conditions and trend filter
# Camarilla levels provide precise intraday support/resistance; EMA50 filters trend direction;
# Volume confirmation reduces false breakouts. Works in bull/bear by only trading with trend.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Midpoint = (R1 + S1)/2 = close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_range = (high_1d_prev - low_1d_prev) * 1.1 / 12.0
    r1 = close_1d_prev + camarilla_range
    s1 = close_1d_prev - camarilla_range
    midpoint = close_1d_prev  # (R1 + S1)/2 simplifies to close
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Align HTF indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        ema_trend = ema_50_12h_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        mid_level = midpoint_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R1 AND close > 12h EMA50 AND volume confirmation
            if price > r1_level and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 AND close < 12h EMA50 AND volume confirmation
            elif price < s1_level and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to midpoint or volume drops
            if price < mid_level or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price reverts to midpoint or volume drops
            if price > mid_level or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals