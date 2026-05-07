#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Camarilla pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous week (standard formula)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    pivot = (w_high + w_low + w_close) / 3
    range_val = w_high - w_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align pivot levels to daily timeframe
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R3 in weekly uptrend with volume
            if close[i] > r3_1d[i] and ema_34_1d[i] > ema_34_1d[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in weekly downtrend with volume
            elif close[i] < s3_1d[i] and ema_34_1d[i] < ema_34_1d[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] < pivot_1d[i] or ema_34_1d[i] < ema_34_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] > pivot_1d[i] or ema_34_1d[i] > ema_34_1d[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts on daily timeframe with weekly trend filter and volume confirmation
# - Camarilla R3/S3 from weekly chart represent strong support/resistance levels from previous week
# - Breakout above R3 in weekly uptrend (EMA34 rising) signals bullish continuation
# - Breakdown below S3 in weekly downtrend (EMA34 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to weekly pivot point or weekly trend reverses
# - Position size 0.25 to manage risk and reduce trade frequency
# - Targets 15-25 trades/year on daily timeframe to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses weekly timeframe for structure and trend, daily for execution timing
# - Designed to work on BTC/ETH as primary targets
# - Focus on high-quality setups to improve generalization to test period (2025-2026)