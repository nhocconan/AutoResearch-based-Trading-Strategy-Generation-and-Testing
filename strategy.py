#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Load 4h data ONCE for trend filter and 1d data for Camarilla pivot
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 4h EMA34 for trend filter (requires close of 4h bar)
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike detection (2x 20-period average on 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(pivot_1h[i]) or np.isnan(ema_34_1h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in 4h uptrend with volume
            if close[i] > r3_1h[i] and ema_34_1h[i] > ema_34_1h[i-1] and vol_condition:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 in 4h downtrend with volume
            elif close[i] < s3_1h[i] and ema_34_1h[i] < ema_34_1h[i-1] and vol_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            if close[i] < pivot_1h[i] or ema_34_1h[i] < ema_34_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            if close[i] > pivot_1h[i] or ema_34_1h[i] > ema_34_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 4h trend filter and volume confirmation on 1h timeframe
# - Camarilla R3/S3 represent strong support/resistance levels from previous day
# - Breakout above R3 in 4h uptrend (EMA34 rising) signals bullish continuation
# - Breakdown below S3 in 4h downtrend (EMA34 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to pivot point or 4h trend reverses
# - Position size 0.20 targets ~15-37 trades/year to avoid fee drag
# - Uses 1d for structure (Camarilla pivot) and 4h for trend filter, 1h for execution timing
# - Designed to work in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
# - Focus on BTC/ETH as primary targets with proper risk control to avoid large drawdowns