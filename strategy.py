#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on 12h with 1w trend filter and volume confirmation.
# Camarilla levels provide institutional support/resistance; weekly trend ensures directional alignment.
# Volume confirms breakout strength. Designed for low-frequency, high-conviction trades in bull/bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # === 1w Camarilla Levels (from daily OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    range_1w = high_1w - low_1w
    camarilla_r3 = close_1w + range_1w * 1.1 / 2
    camarilla_s3 = close_1w - range_1w * 1.1 / 2
    
    # Align to 12h timeframe (wait for weekly close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # === 1w Trend Filter (EMA34) ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume Confirmation (24-period average on 12h) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: Breakout above R3, uptrend, volume
            if breakout_up and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3, downtrend, volume
            elif breakout_down and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Breakdown below S3 or trend reversal
            if breakout_down or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Breakout above R3 or trend reversal
            if breakout_up or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals