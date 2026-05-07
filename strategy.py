#!/usr/bin/env python3
name = "12h_1wPivot_1dTrend_VolumeBreak"
timeframe = "12h"
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
    
    # Weekly high/low/close from daily data (last 5 trading days)
    # Convert daily close to series for rolling
    close_1d = pd.Series(df_1d['close'])
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    
    # Weekly aggregation: 5 days
    weekly_high = high_1d.rolling(window=5, min_periods=5).max().values
    weekly_low = low_1d.rolling(window=5, min_periods=5).min().values
    weekly_close = close_1d.rolling(window=5, min_periods=5).mean().values
    
    # Pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2, 5)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or volume drops
            if close[i] < pp_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or volume drops
            if close[i] > pp_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h weekly pivot breakout with daily trend and volume confirmation
# - Weekly pivot points (S1/R1) act as dynamic support/resistance levels
# - Breakout above S1 with volume in daily uptrend = long opportunity
# - Breakdown below R1 with volume in daily downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to weekly pivot (PP) or volume weakens
# - Position size 0.25 targets 12-37 trades/year, avoiding fee drag
# - Weekly pivot provides structure that works across market regimes
# - Uses 12h timeframe to reduce trade frequency and improve generalization