#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and 2x volume spike confirmation.
# Enter long when price breaks above Camarilla R3 level with 1w EMA50 upward slope and volume > 2x 20-bar average.
# Enter short when price breaks below Camarilla S3 level with 1w EMA50 downward slope and volume confirmation.
# Exit when price retraces to the Camarilla H3/L3 levels respectively.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 30-100 total trades over 4 years (7-25/year).
# Camarilla R3/S3 levels provide good breakout points with sufficient filtering.
# 1w EMA50 slope ensures we trade with the long-term trend.
# Volume spike (2x) filters weak breakouts, reducing false signals.
# Works in both bull and bear markets by following the 1w trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (1-bar change)
    ema_slope = np.diff(ema_1w, prepend=np.nan)
    
    # Align EMA50 and slope to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 1d
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels (R3/S3 are standard breakout points)
    R3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    S3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    H3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    L3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    
    # Volume confirmation: >2x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(H3[i]) or np.isnan(L3[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA50 trend: positive slope = uptrend, negative slope = downtrend
        ema_up = ema_slope_aligned[i] > 0
        ema_down = ema_slope_aligned[i] < 0
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, 1w EMA50 trending up, volume confirm
            if price > R3[i] and ema_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, 1w EMA50 trending down, volume confirm
            elif price < S3[i] and ema_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H3
            if price <= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at L3
            if price >= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals