#!/usr/bin/env python3
# 1h_4h1d_Trend_Follow_With_Volume_Confirmation
# Hypothesis: In 1h timeframe, follow the higher timeframe trend (4h/1d) with volume confirmation
# to avoid false breakouts. Use 4h EMA50 for trend direction and 1d EMA200 for regime filter.
# Enter on pullbacks to 4h EMA20 in direction of higher timeframe trend.
# Long when: 4h trend up (close > EMA50_4h) AND 1d regime bullish (close > EMA200_1d) AND price pulls back to 4h EMA20 from below with volume confirmation.
# Short when: 4h trend down (close < EMA50_4h) AND 1d regime bearish (close < EMA200_1d) AND price pulls back to 4h EMA20 from above with volume confirmation.
# Exit when trend breaks or reverse signal occurs.
# Uses session filter (08-20 UTC) to reduce noise. Position size: 0.20.

name = "1h_4h1d_Trend_Follow_With_Volume_Confirmation"
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
    
    # Get 4h data for trend and entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h EMA20 for entry pullback signals
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA200 for regime filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation (20-period MA on 1h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50_4h (50), EMA20_4h (20), EMA200_1d (200), volume MA (20)
    start_idx = max(50, 20, 200, 20)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend and regime filters
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        bullish_regime = close[i] > ema_200_1d_aligned[i]
        bearish_regime = close[i] < ema_200_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to 4h EMA20 for pullback detection
        if i > 0:
            cross_above_ema20_4h = (close[i] > ema_20_4h_aligned[i]) and (close[i-1] <= ema_20_4h_aligned[i-1])
            cross_below_ema20_4h = (close[i] < ema_20_4h_aligned[i]) and (close[i-1] >= ema_20_4h_aligned[i-1])
        else:
            cross_above_ema20_4h = False
            cross_below_ema20_4h = False
        
        if position == 0:
            # Long entry: 4h uptrend + bullish regime + pullback to EMA20 from below + volume
            if uptrend_4h and bullish_regime and cross_above_ema20_4h and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend + bearish regime + pullback to EMA20 from above + volume
            elif downtrend_4h and bearish_regime and cross_below_ema20_4h and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend breaks, regime turns bearish, or reverse signal
            if not uptrend_4h or not bullish_regime or cross_below_ema20_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h trend breaks, regime turns bullish, or reverse signal
            if not downtrend_4h or not bearish_regime or cross_above_ema20_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals