#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on daily timeframe with weekly trend filter and volume confirmation.
# The weekly trend defines the market bias, while daily Camarilla levels provide precise entry/exit points.
# Volume confirmation ensures breakouts have institutional backing. Designed to work in both bull and bear markets
# by only trading in the direction of the higher timeframe trend.

name = "1D_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Trend (EMA 34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla Levels (from previous day) ===
    # Calculate from previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla multipliers
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        week_trend_up = close[i] > ema_34_1w_aligned[i]
        week_trend_down = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R3 with weekly uptrend and volume
            if close[i] > R3[i] and week_trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with weekly downtrend and volume
            elif close[i] < S3[i] and week_trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Break below S3 or weekly trend turns down
            if close[i] < S3[i] or not week_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R3 or weekly trend turns up
            if close[i] > R3[i] or not week_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals