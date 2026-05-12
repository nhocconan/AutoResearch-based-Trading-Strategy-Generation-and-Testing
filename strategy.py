#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on daily with weekly trend filter and volume confirmation.
# The Camarilla levels provide key support/resistance. Breakout from R3/S3 with
# weekly trend alignment and volume confirmation captures strong moves. Works in
# bull markets via breakouts and in bear via mean reversion at extremes.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Trend Filter (EMA 34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla Levels (based on previous day) ===
    # Calculate using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First day: use current values (will be neutralized by min_periods later)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # R3 and S3 levels
    r3 = prev_close + (range_val * 1.1 / 4)
    s3 = prev_close - (range_val * 1.1 / 4)
    
    # === Volume Confirmation (20-day average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # LONG: Close breaks above R3, weekly uptrend, volume confirmation
            if close[i] > r3[i] and price_above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3, weekly downtrend, volume confirmation
            elif close[i] < s3[i] and price_below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close falls below S3 or trend changes
            if close[i] < s3[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises above R3 or trend changes
            if close[i] > r3[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals