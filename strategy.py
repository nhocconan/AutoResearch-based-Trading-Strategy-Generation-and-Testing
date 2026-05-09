#!/usr/bin/env python3
# Hypothesis: 1h mean reversion with 4h Bollinger Bands and 1d trend filter
# In ranging markets (2025+), price tends to revert to mean at Bollinger Bands
# Long when: price touches lower BB(20,2) on 4h + 1d EMA(50) rising + RSI(14) < 30 on 1h
# Short when: price touches upper BB(20,2) on 4h + 1d EMA(50) falling + RSI(14) > 70 on 1h
# Exit when: price returns to 4h BB middle band or RSI crosses 50
# Position size: 0.20 to limit drawdown. Target: 20-40 trades/year.
# Uses 4h for entry signals, 1d for trend filter, 1h for RSI timing.

name = "1h_BollingerMeanReversion_1dTrend_RSI"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_4h = df_4h['close']
    sma_20_4h = close_4h.rolling(window=20, min_periods=20).mean()
    std_20_4h = close_4h.rolling(window=20, min_periods=20).std()
    bb_upper_4h = sma_20_4h + (2 * std_20_4h)
    bb_lower_4h = sma_20_4h - (2 * std_20_4h)
    bb_middle_4h = sma_20_4h
    
    # Align 4h Bollinger Bands to 1h timeframe
    bb_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_upper_4h.values)
    bb_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_lower_4h.values)
    bb_middle_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_middle_4h.values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_4h_aligned[i]) or np.isnan(bb_lower_4h_aligned[i]) or
            np.isnan(bb_middle_4h_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price at or below 4h lower BB + 1d EMA rising + RSI oversold
            if (close[i] <= bb_lower_4h_aligned[i] and 
                ema_rising_aligned[i] and 
                rsi[i] < 30):
                signals[i] = 0.20
                position = 1
            # Enter short: price at or above 4h upper BB + 1d EMA falling + RSI overbought
            elif (close[i] >= bb_upper_4h_aligned[i] and 
                  ema_falling_aligned[i] and 
                  rsi[i] > 70):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 4h middle BB or RSI crosses above 50
            if (close[i] >= bb_middle_4h_aligned[i]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to 4h middle BB or RSI crosses below 50
            if (close[i] <= bb_middle_4h_aligned[i]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals