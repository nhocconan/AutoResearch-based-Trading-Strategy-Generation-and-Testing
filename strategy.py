#!/usr/bin/env python3
name = "4h_Engulfing_Pattern_1dTrend_Volume"
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
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Bullish/Bearish Engulfing detection
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close > open_price.shift(1)) & (open_price < close.shift(1)) & \
                     (close > open_price.shift(1)) & (open_price < close.shift(1))
    # Actually: Bullish engulf: current green candle engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price <= close.shift(1)) & (close >= open_price.shift(1)) & \
                     (close > open_price) & (open_price < close.shift(1))
    # Bearish engulf: current red candle engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price >= close.shift(1)) & (close <= open_price.shift(1)) & \
                     (close < open_price) & (open_price > close.shift(1))
    
    # Simplified correct engulfing
    bullish_engulf = (close > open_price) & (open_price <= close.shift(1)) & (close >= open_price.shift(1)) & (close > open_price.shift(1))
    bearish_engulf = (close < open_price) & (open_price >= close.shift(1)) & (close <= open_price.shift(1)) & (close < open_price.shift(1))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure volume avg has enough data
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish engulf + 1d uptrend + volume filter
            if bullish_engulf[i] and (close[i] > ema34_1d_aligned[i]) and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulf + 1d downtrend + volume filter
            elif bearish_engulf[i] and (close[i] < ema34_1d_aligned[i]) and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish engulf or trend reversal
            if bearish_engulf[i] or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish engulf or trend reversal
            if bullish_engulf[i] or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals