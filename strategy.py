#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume_v1"
timeframe = "1h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Align daily levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h timeframe for signal direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume spike detection: 12-period average (6 hours)
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 21, 12)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(vol_ma_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily/4h uptrend
            vol_condition = volume[i] > vol_ma_12[i] * 2.0
            daily_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            fourh_uptrend = ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and daily_uptrend and fourh_uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and daily/4h downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not daily_uptrend and not fourh_uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_12[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_12[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with daily and 4h trend + volume confirmation
# - Daily Camarilla S1/R1 act as strong support/resistance levels
# - Breakout above S1 with volume in daily and 4h uptrend = long opportunity
# - Breakdown below R1 with volume in daily and 4h downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.20 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual daily and 4h data for proper multi-timeframe alignment
# - Strict conditions to prevent overtrading while maintaining edge in BTC/ETH
# - Added 4h trend filter to improve signal quality and reduce false breakouts