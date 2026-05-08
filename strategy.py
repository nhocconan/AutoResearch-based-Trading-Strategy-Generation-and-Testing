#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elliott Wave-inspired retracement with 1d trend filter.
# Long when: 6h price retraces to 0.618 Fibonacci level of prior swing low AND 1d EMA50 > EMA200 (uptrend) AND 6h volume > 1.5x 20-period average.
# Short when: 6h price retraces to 0.382 Fibonacci level of prior swing high AND 1d EMA50 < EMA200 (downtrend) AND 6h volume > 1.5x 20-period average.
# Exit when price crosses the opposite Fibonacci level or 6h EMA21 cross.
# Uses Fibonacci retracements for mean reversion in trending markets with volume confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_FibRetrace_1dEMA_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h swing points for Fibonacci levels (5-period lookback)
    swing_high = pd.Series(high).rolling(window=5, center=False).max().values
    swing_low = pd.Series(low).rolling(window=5, center=False).min().values
    
    # 6h Fibonacci retracement levels
    diff = swing_high - swing_low
    fib_0618 = swing_low + 0.618 * diff  # 61.8% retracement (long entry)
    fib_0382 = swing_low + 0.382 * diff  # 38.2% retracement (short entry)
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 6h EMA21 for exit
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(fib_0618[i]) or np.isnan(fib_0382[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at 0.618 fib, 1d uptrend, volume spike
            near_fib_long = abs(close[i] - fib_0618[i]) < (0.005 * close[i])  # Within 0.5% of fib level
            uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
            if near_fib_long and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: price at 0.382 fib, 1d downtrend, volume spike
            near_fib_short = abs(close[i] - fib_0382[i]) < (0.005 * close[i])  # Within 0.5% of fib level
            downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
            if near_fib_short and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 0.382 fib OR EMA21 cross down
            exit_long = (close[i] < fib_0382[i]) or (close[i] < ema21[i] and ema21[i-1] >= close[i-1])
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 0.618 fib OR EMA21 cross up
            exit_short = (close[i] > fib_0618[i]) or (close[i] > ema21[i] and ema21[i-1] <= close[i-1])
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals