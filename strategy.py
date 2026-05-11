#!/usr/bin/env python3
# 1d_Camarilla_R3S3_Breakout_WeeklyTrend
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries with weekly trend filter (EMA34 on weekly chart) and volume confirmation. Enters long when price breaks above R3 with weekly uptrend and volume spike; enters short when price breaks below S3 with weekly downtrend and volume spike. Exits when price returns to the weekly EMA34 or trend reverses. Designed for daily timeframe to limit trades (7-25/year) and avoid fee drag. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets.

name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng / 4
    s3 = close_1d - 1.1 * rng / 4
    
    # --- Weekly EMA34 for trend direction ---
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_slope = ema_34_1w - np.roll(ema_34_1w, 1)
    ema_34_1w_slope[0] = 0
    ema_34_1w_slope = pd.Series(ema_34_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    
    # --- Daily volume confirmation (volume > 20-period average) ---
    vol_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_34_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_slope)
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for daily EMA20 (20) and weekly EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(ema_34_1w_slope_aligned[i]) or
            np.isnan(vol_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from weekly EMA34 slope
        uptrend = ema_34_1w_slope_aligned[i] > 0
        downtrend = ema_34_1w_slope_aligned[i] < 0
        
        # Volume spike condition
        vol_spike = volume[i] > vol_20_1d_aligned[i] * 1.5  # 50% above average
        
        if position == 0:
            if uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + price above R3
                if close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + price below S3
                if close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to weekly EMA34 OR trend turns down
                if close[i] < ema_34_1w_aligned[i] or downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly EMA34 OR trend turns up
                if close[i] > ema_34_1w_aligned[i] or uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals