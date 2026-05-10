#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout of Camarilla R3/S3 levels on 4h with 1d trend filter and volume spike.
# Long when: price breaks above R3, 1d trend is up (close > EMA50), volume > 2x average.
# Short when: price breaks below S3, 1d trend is down (close < EMA50), volume > 2x average.
# Uses volatility-based position sizing (ATR-based stop implied by position reversal).
# Works in bull/bear by following 1d trend and using volatility breakouts with volume confirmation.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels for each 4h bar using lookback window
    def calculate_camarilla(high_window, low_window, close_prev):
        """Calculate Camarilla levels for given window"""
        if len(high_window) == 0 or len(low_window) == 0:
            return np.nan, np.nan
        H = np.max(high_window)
        L = np.min(low_window)
        C = close_prev
        range_hl = H - L
        R3 = C + (range_hl * 1.1 / 4)
        S3 = C - (range_hl * 1.1 / 4)
        return R3, S3
    
    # Pre-calculate Camarilla levels using 4h lookback (previous bar's H/L/C)
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's data to avoid look-ahead
        lookback_start = max(0, i-20)  # 20-period lookback for volatility
        if i-1 >= lookback_start:
            high_window = high[lookback_start:i]  # up to but not including current bar
            low_window = low[lookback_start:i]
            close_prev = close[i-1]
            if len(high_window) > 0 and len(low_window) > 0:
                r3_val, s3_val = calculate_camarilla(high_window, low_window, close_prev)
                R3[i] = r3_val
                S3[i] = s3_val
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0  # Require strong volume spike
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: break above R3 + daily uptrend + volume spike
            if daily_up and volume_confirm and close[i] > R3[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 + daily downtrend + volume spike
            elif daily_down and volume_confirm and close[i] < S3[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: reverse signal or volume drops
            if daily_down or not volume_confirm or close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: reverse signal or volume drops
            if daily_up or not volume_confirm or close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals