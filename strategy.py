#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: Buy breakouts above Donchian(20) high with volume > 1.5x 20-period average in uptrend (EMA50 > EMA200).
# Sell breakdowns below Donchian(20) low with volume > 1.5x 20-period average in downtrend (EMA50 < EMA200).
# Works in bull markets via breakout momentum and in bear markets via breakdown continuation.
# Volume confirmation filters false breakouts; trend filter avoids counter-trend whipsaws.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1D_TREND_FILTER"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mdata import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 and EMA200 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 4h volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_avg[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, lookback - 1)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend determination
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # LONG: Donchian breakout + volume + uptrend
            if close[i] > highest_high[i] and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown + volume + downtrend
            elif close[i] < lowest_low[i] and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Donchian breakdown or trend reversal
            if close[i] < lowest_low[i] or ema50_1d_aligned[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Donchian breakout or trend reversal
            if close[i] > highest_high[i] or ema50_1d_aligned[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals