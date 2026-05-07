#!/usr/bin/env python3
name = "4h_1d_PivotBreakout_RangeFilter_Volume"
timeframe = "4h"
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
    
    # Daily Pivot (standard) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Daily Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align daily levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily RSI(14) for range filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume, uptrend, and not overbought
            vol_condition = volume[i] > vol_ma_6[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            not_overbought = rsi_14_1d_aligned[i] < 70
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend and not_overbought:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume, downtrend, and not oversold
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend and rsi_14_1d_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or momentum fades
            if close[i] < s1_aligned[i] or rsi_14_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or momentum fades
            if close[i] > r1_aligned[i] or rsi_14_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Daily Pivot S1/R1 breakout with 1d trend filter and range filter
# - Daily Pivot S1/R1 act as key support/resistance levels from prior day
# - Breakout above S1 with volume in daily uptrend (not overbought) = long opportunity
# - Breakdown below R1 with volume in daily downtrend (not oversold) = short opportunity
# - Volume spike (1.5x average) confirms institutional participation
# - RSI filter (30-70) prevents entries in extreme conditions, reducing whipsaws
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or RSI reaches extreme levels
# - Position size 0.25 targets ~30-60 trades/year, avoiding fee drag
# - Uses actual daily Pivot levels for stability and relevance
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Range filter avoids overextended moves
# - Volume confirmation reduces false breakouts
# - Designed for 4h timeframe with daily context for better signal quality
# - Aims for 120-240 total trades over 4 years (30-60/year) to stay within limits
# - Focus on BTC/ETH as primary targets with volume and trend confirmation