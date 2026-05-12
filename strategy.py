#!/usr/bin/env python3
name = "1d_Camarilla_R3_S4_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 50-period EMA for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === DAILY CAMARILLA PIVOTS (based on previous day) ===
    # Calculate pivot levels using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day: use first available values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.0833)
    r2 = pivot + (range_hl * 1.1666)
    r3 = pivot + (range_hl * 1.2500)
    r4 = pivot + (range_hl * 1.3333)
    
    # Support levels
    s1 = pivot - (range_hl * 1.0833)
    s2 = pivot - (range_hl * 1.1666)
    s3 = pivot - (range_hl * 1.2500)
    s4 = pivot - (range_hl * 1.3333)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and pivots
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and weekly uptrend
            if (close[i] > r3[i] and 
                volume_spike[i] and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and weekly downtrend
            elif (close[i] < s3[i] and 
                  volume_spike[i] and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below R1 or weekly trend turns down
            if close[i] < r1[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S1 or weekly trend turns up
            if close[i] > s1[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals