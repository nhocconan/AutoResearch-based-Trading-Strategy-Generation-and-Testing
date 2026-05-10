#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Buy at Camarilla S3 and sell at R3 on 12h timeframe with 1d trend filter and volume confirmation.
# Works in bull/bear by following daily trend and using Camarilla levels as reversal points in ranging markets.
# Volume confirmation ensures institutional participation. Target: 15-30 trades/year per symbol.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (24-period = 12 days)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA50 for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Calculate Camarilla levels for previous day
    camarilla_S3 = np.zeros(len(close_1d))
    camarilla_R3 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_val = prev_high - prev_low
        camarilla_S3[i] = prev_close - range_val * 1.1 / 6
        camarilla_R3[i] = prev_close + range_val * 1.1 / 6
    
    # Align daily data to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(daily_uptrend_aligned[i]) or 
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        camarilla_S3_val = camarilla_S3_aligned[i]
        camarilla_R3_val = camarilla_R3_aligned[i]
        
        if position == 0:
            # Enter long: daily uptrend + price at S3 + volume confirmation
            if daily_up and volume_confirm:
                if close[i] <= camarilla_S3_val * 1.005 and close[i] >= camarilla_S3_val * 0.995:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + price at R3 + volume confirmation
            elif daily_down and volume_confirm:
                if close[i] >= camarilla_R3_val * 0.995 and close[i] <= camarilla_R3_val * 1.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: trend reverses or price reaches R3 (take profit)
            if not daily_up or close[i] >= camarilla_R3_val * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reverses or price reaches S3 (take profit)
            if not daily_down or close[i] <= camarilla_S3_val * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals