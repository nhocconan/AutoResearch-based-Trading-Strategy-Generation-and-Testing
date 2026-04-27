#!/usr/bin/env python3
"""
6h_Daily_Engulfing_with_Weekly_Trend_and_Volume
Hypothesis: On 6h timeframe, enter long when a bullish engulfing candle forms above the 200-period EMA, 
and short when a bearish engulfing candle forms below the 200-period EMA, only when the weekly trend 
agrees (price above/below weekly EMA20) and volume confirms (>1.5x 20-period average). 
Exit on opposite engulfing signal or trend failure. 
This captures momentum continuation in trending markets while avoiding counter-trend noise. 
Designed for 6-12 trades per year per symbol, suitable for 6h timeframe.
"""

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
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Bullish engulfing: current bullish candle engulfs previous bearish candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close) & \
                        (close > open_price.shift(1)) & (open_price < close.shift(1)) & \
                        (close > open_price.shift(1)) & (open_price < close.shift(1))
    # Simplified: current candle bullish and engulfs previous candle's body
    bullish_engulfing = (close > open_price) & (open_price <= close.shift(1)) & (close >= open_price.shift(1)) & \
                        ((close - open_price) > (open_price.shift(1) - close.shift(1)))
    
    # Bearish engulfing: current bearish candle engulfs previous bullish candle
    bearish_engulfing = (close < open_price) & (open_price >= close.shift(1)) & (close <= open_price.shift(1)) & \
                        ((open_price - close) > (close.shift(1) - open_price.shift(1)))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for indicators
    start_idx = 200  # need 200 for daily EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish engulfing above daily EMA200, weekly uptrend, volume confirmation
            if (bullish_engulfing[i] and 
                close[i] > ema200_1d_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing below daily EMA200, weekly downtrend, volume confirmation
            elif (bearish_engulfing[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish engulfing or price below daily EMA200 or weekly downtrend
            if (bearish_engulfing[i] or 
                close[i] < ema200_1d_aligned[i] or 
                close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish engulfing or price above daily EMA200 or weekly uptrend
            if (bullish_engulfing[i] or 
                close[i] > ema200_1d_aligned[i] or 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Daily_Engulfing_with_Weekly_Trend_and_Volume"
timeframe = "6h"
leverage = 1.0