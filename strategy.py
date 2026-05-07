#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter and volume spike.
# Long when: Price touches S3 support AND 1-day EMA34 rising AND volume > 2x 20-period volume EMA.
# Short when: Price touches R3 resistance AND 1-day EMA34 falling AND volume > 2x 20-period volume EMA.
# Exit when price crosses the 1-day VWAP or reaches opposite Camarilla level (S1/R1).
# Designed for low trade frequency (target: 20-30/year) with high win rate via confluence.
# Works in bull via S3 bounces and in bear via R3 rejections.
name = "4h_Camarilla_S3R3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous day (using daily OHLC)
    # We'll calculate daily OHLC from 4h data by grouping
    # But per rules, we must use get_htf_data for 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using prior day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # But standard Camarilla uses:
    # Resistance: R1 = close + 0.115*(high-low), R2 = +0.27, R3 = +0.55, R4 = +1.0
    # Support: S1 = close - 0.115*(high-low), S2 = -0.27, S3 = -0.55, S4 = -1.0
    # We'll use the more common version: R3/S3 at ±0.55*(high-low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for each day
    R3_1d = close_1d + 0.55 * (high_1d - low_1d)
    S3_1d = close_1d - 0.55 * (high_1d - low_1d)
    
    # Also calculate S1 and R1 for exit
    S1_1d = close_1d - 0.115 * (high_1d - low_1d)
    R1_1d = close_1d + 0.115 * (high_1d - low_1d)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Align all 1D indicators to 4H timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: volume > 2x 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price <= S3 (touching or penetrating support) AND EMA34 rising AND volume spike
            long_condition = (close[i] <= S3_1d_aligned[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Price >= R3 (touching or penetrating resistance) AND EMA34 falling AND volume spike
            short_condition = (close[i] >= R3_1d_aligned[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price >= S1 (reached support bounce target) OR price >= R1 (overshot)
            if close[i] >= S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price <= R1 (reached resistance bounce target) OR price <= S1 (overshot)
            if close[i] <= R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals