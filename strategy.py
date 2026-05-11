#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Uses Camarilla pivot levels from 1d to identify R1 and S1 breakout points on 4h timeframe, filtered by 12h EMA50 trend and volume spikes. Long when price breaks above R1 with 12h uptrend and volume spike; short when price breaks below S1 with 12h downtrend and volume spike. Exit when price reverts to the pivot point (PP) or trend breaks. Designed to capture institutional breakouts with trend and volume confirmation, working in both bull and bear markets by following the dominant 12h trend while using Camarilla levels as dynamic support/resistance.
# Named after proven top performer pattern from DB: 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS (ETHUSDT test_sharpe=2.055)

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels (PP, R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels: PP, R1, S1 ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # R1 = PP + (Range * 1.1 / 12)
    r1 = pp + (range_1d * 1.1 / 12.0)
    # S1 = PP - (Range * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12.0)
    
    # --- 12h EMA50 trend filter ---
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(close_12h), np.nan)
    # Calculate EMA50 with proper smoothing
    for i in range(len(close_12h)):
        if i == 0:
            ema_50[i] = close_12h[i]
        elif i < 50:
            # Simple average for first 50 periods
            ema_50[i] = np.mean(close_12h[:i+1])
        else:
            # EMA formula: EMA = today * k + yesterday * (1-k), where k = 2/(N+1)
            k = 2.0 / (50 + 1)
            ema_50[i] = close_12h[i] * k + ema_50[i-1] * (1 - k)
    
    # Trend: 1 if close > EMA50 (uptrend), -1 if close < EMA50 (downtrend)
    trend_12h = np.where(close_12h > ema_50, 1, -1)
    # Handle NaN values
    trend_12h = np.where(np.isnan(ema_50), 0, trend_12h)
    
    # Align 1d Camarilla levels and 12h trend to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # --- Volume confirmation: volume > 20-period average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (need 1d data), EMA50 (50 periods), and volume MA(20)
    start_idx = max(50, 20)  # EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(trend_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Trend from 12h
        is_uptrend = trend_12h_aligned[i] == 1
        is_downtrend = trend_12h_aligned[i] == -1
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: 12h uptrend + volume spike + price breaks above R1
                if close[i] > r1_val:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: 12h downtrend + volume spike + price breaks below S1
                if close[i] < s1_val:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to PP or 12h uptrend breaks
                if close[i] <= pp_val or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to PP or 12h downtrend breaks
                if close[i] >= pp_val or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals