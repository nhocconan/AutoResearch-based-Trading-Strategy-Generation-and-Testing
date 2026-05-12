#!/usr/bin/env python3
name = "6h_Volume_Spice_Trend_Filter"
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
    
    # Daily trend: EMA34 on 1d close (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma)
    
    # Price momentum: bullish/bearish candle
    bullish_candle = close > open_
    bearish_candle = close < open_
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # ensure EMA34 has enough data
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily EMA34 + bullish candle + volume spike
            if (close[i] > ema34_1d_aligned[i]) and bullish_candle[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 + bearish candle + volume spike
            elif (close[i] < ema34_1d_aligned[i]) and bearish_candle[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA34
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA34
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals