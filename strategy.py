#!/usr/bin/env python3
"""
6h_Heikin_Ashi_Engulfing_12hTrend_Volume
Hypothesis: On 6h timeframe, Heikin Ashi bullish/bearish engulfing candles combined with 12h EMA50 trend filter and volume confirmation capture high-probability momentum reversals. The Heikin Ashi filter reduces noise and false signals, while the 12h trend filter ensures alignment with higher timeframe momentum, making it effective in both bull and bear markets.
"""
name = "6h_Heikin_Ashi_Engulfing_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Heikin Ashi candles
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
    
    # Heikin Ashi engulfing signals
    bullish_engulf = (ha_close > ha_open) & (ha_close > ha_open) & \
                     (ha_close > ha_open) & (ha_open < ha_close) & \
                     (ha_close > ha_open) & (ha_open < ha_close) & \
                     (ha_close[i-1] < ha_open[i-1]) & (ha_close > ha_open) & \
                     (ha_open < ha_close) & (ha_close > ha_open) & \
                     (ha_open < ha_close)  # Simplified: current bullish engulfing previous bearish
    bearish_engulf = (ha_close < ha_open) & (ha_close < ha_open) & \
                     (ha_close < ha_open) & (ha_open > ha_close) & \
                     (ha_close < ha_open) & (ha_open > ha_close) & \
                     (ha_close[i-1] > ha_open[i-1]) & (ha_close < ha_open) & \
                     (ha_open > ha_close) & (ha_close < ha_open) & \
                     (ha_open > ha_close)  # Simplified: current bearish engulfing previous bullish
    
    # Correct vectorized engulfing calculation
    bullish_engulf = (ha_close > ha_open) & (ha_close > ha_open) & \
                     (ha_close > ha_open) & (ha_open < ha_close) & \
                     (ha_close > ha_open) & (ha_open < ha_close) & \
                     (ha_close[1:] < ha_open[:-1]) & (ha_close[:-1] < ha_open[:-1])  # Placeholder
    
    # Proper implementation
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing: current green candle completely engulfs previous red candle
        if ha_close[i] > ha_open[i] and ha_close[i-1] < ha_open[i-1]:
            if ha_close[i] >= ha_open[i-1] and ha_open[i] <= ha_close[i-1]:
                bullish_engulf[i] = True
        # Bearish engulfing: current red candle completely engulfs previous green candle
        if ha_close[i] < ha_open[i] and ha_close[i-1] > ha_open[i-1]:
            if ha_close[i] <= ha_open[i-1] and ha_open[i] >= ha_close[i-1]:
                bearish_engulf[i] = True
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing + 12h uptrend + volume
            if bullish_engulf[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing + 12h downtrend + volume
            elif bearish_engulf[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: opposite engulfing signal or trend failure
            if position == 1:
                if bearish_engulf[i] or close[i] < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bullish_engulf[i] or close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals