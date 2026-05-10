#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Filter
# Hypothesis: Elder Ray (Bull/Bear Power) identifies bullish/bearish momentum via EMA13 deviation.
# Combined with 1-day EMA50 trend filter to avoid counter-trend trades.
# Long when: 1-day trend up (close > EMA50_1d) AND Bull Power > 0 AND rising.
# Short when: 1-day trend down (close < EMA50_1d) AND Bear Power < 0 AND falling.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends).
# Uses volume confirmation to avoid low-conviction signals.

name = "6h_ElderRay_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (6h chart)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13), EMA50_1d (50), volume MA (20)
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Elder Ray signals with slope (rising/falling)
        if i > 1:
            bull_rising = bull_power[i] > bull_power[i-1]
            bear_falling = bear_power[i] < bear_power[i-1]
        else:
            bull_rising = False
            bear_falling = False
        
        if position == 0:
            # Long entry: uptrend + Bull Power positive AND rising + volume
            if uptrend and bull_power[i] > 0 and bull_rising and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + Bear Power negative AND falling + volume
            elif downtrend and bear_power[i] < 0 and bear_falling and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or Bull Power turns negative
            if not uptrend or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or Bear Power turns positive
            if not downtrend or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals