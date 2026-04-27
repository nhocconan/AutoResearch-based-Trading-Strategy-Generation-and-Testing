#!/usr/bin/env python3
"""
1d_Keltner_UpperBand_Touch_1wTrend_Filter
Hypothesis: On daily timeframe, go long when price touches upper Keltner band (EMA20 + 2*ATR) only when weekly EMA50 is rising (bullish trend), and short when price touches lower Keltner band (EMA20 - 2*ATR) only when weekly EMA50 is falling (bearish trend). Exit when price crosses back below/above the EMA20 middle band. Uses volume confirmation (>1.5x 20-day average) to avoid false breakouts. Designed to capture trend continuations in both bull and bear markets with low trade frequency.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter (rising/falling)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = np.nan
    weekly_trend_up = ema_50_1w > ema_50_1w_prev
    weekly_trend_down = ema_50_1w < ema_50_1w_prev
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Daily EMA20 for Keltner middle band
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(10) for Keltner bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    keltner_upper = ema_20 + 2.0 * atr_10
    keltner_lower = ema_20 - 2.0 * atr_10
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA, daily EMA/ATR, volume
    start_idx = max(50, 20, 10, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        keltner_up = keltner_upper[i]
        keltner_low = keltner_lower[i]
        ema_20_val = ema_20[i]
        vol_conf = volume_confirm[i]
        weekly_up = weekly_trend_up_aligned[i]
        weekly_down = weekly_trend_down_aligned[i]
        
        if position == 0:
            # Long: price touches upper Keltner band with volume confirmation AND weekly uptrend
            if close[i] >= keltner_up and vol_conf and weekly_up:
                signals[i] = size
                position = 1
            # Short: price touches lower Keltner band with volume confirmation AND weekly downtrend
            elif close[i] <= keltner_low and vol_conf and weekly_down:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses back below EMA20 (middle band)
            if close[i] < ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above EMA20 (middle band)
            if close[i] > ema_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_UpperBand_Touch_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0