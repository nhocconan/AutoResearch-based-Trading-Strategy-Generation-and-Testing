#!/usr/bin/env python3
# 1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_VOLUME_CONFIRMATION
# Hypothesis: Camarilla pivot breakout on 1h with 4h trend filter and volume confirmation.
# Long when price breaks above R1 with 4h uptrend and volume > 1.5x average; short when price breaks below S1 with 4h downtrend and volume > 1.5x average.
# Exit when price returns to the pivot point. Designed to capture breakouts with trend and volume confirmation to avoid false signals.
# Targets 15-35 trades/year to minimize fee drag while capturing high-probability breakouts in both bull and bear markets.

name = "1H_CAMARILLA_R1_S1_BREAKOUT_4HTREND_VOLUME_CONFIRMATION"
timeframe = "1h"
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
    
    # Camarilla pivot points for previous day (using previous day's high, low, close)
    # We'll use rolling window of 24h (96 bars for 15m, but for 1h we use 24 bars)
    # Since we're on 1h timeframe, we need daily high/low/close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Align to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rang = prev_high_aligned - prev_low_aligned
    r1 = prev_close_aligned + rang * 1.1 / 12
    s1 = prev_close_aligned - rang * 1.1 / 12
    pivot = prev_close_aligned  # Camarilla pivot is close
    
    # 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema4h_aligned = align_htf_to_ltf(prices, df_4h, ema4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: price breaks above R1 with 4h uptrend and volume confirmation
            if close[i] > r1[i] and close[i-1] <= r1[i-1] and ema4h_aligned[i] > ema4h_aligned[i-1] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 with 4h downtrend and volume confirmation
            elif close[i] < s1[i] and close[i-1] >= s1[i-1] and ema4h_aligned[i] < ema4h_aligned[i-1] and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot point
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price returns to pivot point
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals