#!/usr/bin/env python3
name = "4h_Bullish_Engulfing_1dTrend_Volume"
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
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bullish engulfing pattern: current candle engulfs previous candle
    bullish_engulfing = (close[i] > open_price[i-1]) & (open_price[i] < close[i-1])
    bearish_engulfing = (close[i] < open_price[i-1]) & (open_price[i] > close[i-1])
    
    # Volume filter: current volume > 1.5x average of last 20 periods
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Wait for EMA and volume
    
    for i in range(start_idx, n):
        # Check if we have valid data
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing + uptrend (price > EMA) + volume
            if (bullish_engulfing[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing + downtrend (price < EMA) + volume
            elif (bearish_engulfing[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close below EMA (trend change) or bearish engulfing
            if close[i] < ema_34_1d_aligned[i] or bearish_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close above EMA (trend change) or bullish engulfing
            if close[i] > ema_34_1d_aligned[i] or bullish_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h bullish/bearish engulfing patterns with 1d EMA trend filter and volume confirmation.
# Engulfing candles signal strong momentum shifts. Combined with 1d trend (price above/below EMA34),
# they capture trend continuations with high probability. Volume filter ensures institutional participation.
# Works in both bull and bear markets by following the 1d trend. Target: 20-40 trades/year.
# Position size 0.25 limits risk during drawdowns. Engulfing patterns are rare enough to avoid overtrading.