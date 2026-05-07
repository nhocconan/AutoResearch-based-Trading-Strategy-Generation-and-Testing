#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Weekly Camarilla pivot breakouts on 1d capture multi-week trends.
# Uses weekly Camarilla R3/S3 levels for entry, 1w EMA34 trend filter, and volume spike confirmation.
# Works in bull markets via long breakouts above weekly R3 and bear via short breakdowns below weekly S3.
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly Camarilla levels (based on previous week's range)
    # Calculate pivot and levels from previous week's OHLC
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (wk_high + wk_low + wk_close) / 3.0
    # Range = H - L
    range_wk = wk_high - wk_low
    # Camarilla levels
    r3 = pivot + range_wk * 1.1 / 2  # R3 = pivot + (range * 1.1/2)
    s3 = pivot - range_wk * 1.1 / 2  # S3 = pivot - (range * 1.1/2)
    
    # Align weekly levels to daily (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(wk_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 34 periods for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above weekly R3 or below weekly S3
        breakout_up = close[i] > r3_aligned[i]
        breakout_down = close[i] < s3_aligned[i]
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: upward breakout above R3 + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout below S3 + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below weekly S3 or trend reversal
            if close[i] < s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above weekly R3 or trend reversal
            if close[i] > r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals