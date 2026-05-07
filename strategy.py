#!/usr/bin/env python3
name = "1d_1w_PivotBreakout_TrendVolume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Pivot (standard) from previous day
    prev_high = high[:-1]  # shift(1)
    prev_low = low[:-1]
    prev_close = close[:-1]
    
    # Pad with NaN for first day
    prev_high = np.concatenate([[np.nan], prev_high])
    prev_low = np.concatenate([[np.nan], prev_low])
    prev_close = np.concatenate([[np.nan], prev_close])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Volume spike detection: 20-day average (approximately 1 month)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(s1[i]) or 
            np.isnan(r1[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > s1[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 1d Pivot S1/R1 breakout with weekly trend and volume confirmation
# - Daily Pivot S1/R1 act as key support/resistance levels from previous day
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x 20-day average) confirms institutional participation
# - Weekly EMA(34) trend filter ensures alignment with higher timeframe momentum
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.30 targets ~10-25 trades/year, avoiding fee drag
# - Uses daily Pivot levels for responsiveness and weekly trend for filtering
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Pivot (1d) + trend (1w) + volume (1d) not recently tried on 1d
# - Aims for 40-100 total trades over 4 years (10-25/year) to stay within limits
# - Focus on BTC/ETH as primary targets, avoids SOL-only bias