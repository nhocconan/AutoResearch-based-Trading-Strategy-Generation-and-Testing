#!/usr/bin/env python3
name = "6h_1d_PivotReversal_TrendConfirm_v1"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop for Pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Pivot (standard) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align daily levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 12h EMA(34) for trend filter (1d equivalent)
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1 with bullish momentum in uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            price_near_s1 = close[i] <= s1_aligned[i] * 1.005  # Within 0.5% of S1
            price_above_prev_low = close[i] > low[i-1]  # Higher low
            
            if price_near_s1 and vol_condition and uptrend and price_above_prev_low:
                signals[i] = 0.25
                position = 1
            # Short: price near R1 with bearish momentum in downtrend
            elif close[i] >= r1_aligned[i] * 0.995 and vol_condition and not uptrend and close[i] < high[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks above R1 or momentum fails
            if close[i] > r1_aligned[i] * 1.005 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks below S1 or momentum fails
            if close[i] < s1_aligned[i] * 0.995 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Pivot S1/R1 reversal with 12h trend and volume confirmation
# - Daily Pivot S1/R1 act as key support/resistance from prior session
# - Long when price approaches S1 with bullish signals in 12h uptrend
# - Short when price approaches R1 with bearish signals in 12h downtrend
# - Volume spike (1.8x average) confirms institutional participation at key levels
# - Price action filters: higher low for longs, lower high for shorts
# - Works in both bull (buy S1 bounces in uptrend) and bear (sell R1 bounces in downtrend)
# - Exit when price breaks opposite level or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Pivot levels for better responsiveness vs weekly
# - 12h trend filter reduces whipsaws vs same timeframe
# - Novel: Pivot reversal (not breakout) with momentum confirmation on 6h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits