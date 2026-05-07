# 6h_1d_1w_Camarilla_S1R1_Breakout_Trend_Filter
# Hypothesis: 6h Camarilla S1/R1 breakout with weekly trend filter and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Weekly trend filter (EMA 50) ensures trades align with higher timeframe momentum
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull and bear markets via weekly trend filter
# - Entry: Price breaks S1/R1 with volume and weekly trend alignment
# - Exit: Price returns to S1/R1 or volume drops below average
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels for better responsiveness
# - Weekly trend filter reduces false signals in choppy markets

name = "6h_1d_1w_Camarilla_S1R1_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    
    # Align daily levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Weekly trend filter: EMA(50) on weekly close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla S1/R1 breakout with weekly trend filter and volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Weekly trend filter (EMA 50) ensures trades align with higher timeframe momentum
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull and bear markets via weekly trend filter
# - Entry: Price breaks S1/R1 with volume and weekly trend alignment
# - Exit: Price returns to S1/R1 or volume drops below average
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels for better responsiveness
# - Weekly trend filter reduces false signals in choppy markets