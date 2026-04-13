#!/usr/bin/env python3
"""
4h_1D_Turtle_Soup_Reversal_v1
Hypothesis: Buy when price fails to break below previous day's low (bull trap) with bullish engulfing candle and volume > 1.5x average. Sell when price fails to break above previous day's high (bear trap) with bearish engulfing candle and volume confirmation. Uses 4h timeframe with 1-day trend filter (price > EMA50 for longs, < EMA50 for shorts). Designed to work in both bull and bear markets by trapping false breakouts at key daily levels. Low frequency (~25-35 trades/year) with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Previous day's high/low/close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Align daily levels to 4h timeframe (wait for daily close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Daily EMA50 trend filter
    ema50_1d_raw = pd.Series(prev_close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_raw)
    
    # Candlestick patterns
    bullish_engulfing = (close[i] > open_price[i]) & (open_price[i] < close[i-1]) & (close[i] > close[i-1])
    bearish_engulfing = (close[i] < open_price[i]) & (open_price[i] > close[i-1]) & (close[i] < close[i-1])
    # Compute arrays for patterns
    bullish_engulfing_arr = np.zeros(n, dtype=bool)
    bearish_engulfing_arr = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bullish_engulfing_arr[i] = (close[i] > open_price[i]) and (open_price[i] < close[i-1]) and (close[i] > close[i-1])
        bearish_engulfing_arr[i] = (close[i] < open_price[i]) and (open_price[i] > close[i-1]) and (close[i] < close[i-1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: bull trap - price closes back above prev day's low after probing below
        long_signal = (low[i] < prev_low_aligned[i] and  # probed below low
                      close[i] > prev_low_aligned[i] and  # closed back above
                      bullish_engulfing_arr[i] and
                      volume_expansion[i] and
                      close[i] > ema50_1d_aligned[i])  # above daily EMA50
        
        # Short signal: bear trap - price closes back below prev day's high after probing above
        short_signal = (high[i] > prev_high_aligned[i] and  # probed above high
                       close[i] < prev_high_aligned[i] and  # closed back below
                       bearish_engulfing_arr[i] and
                       volume_expansion[i] and
                       close[i] < ema50_1d_aligned[i])  # below daily EMA50
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 12 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 12 and long_signal:
            position = 1
            signals[i] = position_size
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1D_Turtle_Soup_Reversal_v1"
timeframe = "4h"
leverage = 1.0