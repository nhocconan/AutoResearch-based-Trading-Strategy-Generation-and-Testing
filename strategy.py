#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Moving Average Crossover with 1-day Volume Confirmation and RSI Filter
# Long when fast MA crosses above slow MA AND 1-day RSI > 50 AND volume > 1.5x 20-period average
# Short when fast MA crosses below slow MA AND 1-day RSI < 50 AND volume > 1.5x 20-period average
# Exit when opposite crossover occurs
# Uses MA crossover for trend following, RSI for trend strength on higher timeframe, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Moving Averages on 4h (9-period fast, 21-period slow)
    close_series = pd.Series(close)
    ma_fast = close_series.rolling(window=9, min_periods=9).mean()
    ma_slow = close_series.rolling(window=21, min_periods=21).mean()
    
    # Calculate RSI on 1d (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi_1d = (100 - (100 / (1 + rs))).values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 21 for MA + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_current = rsi_1d[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Get RSI values aligned to 4h timeframe
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
        rsi_current_aligned = rsi_1d_aligned[i]
        
        if position == 0:
            # Long setup: fast MA crosses above slow MA + RSI > 50 (bullish) + volume confirmation
            if (ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1] and 
                rsi_current_aligned > 50 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: fast MA crosses below slow MA + RSI < 50 (bearish) + volume confirmation
            elif (ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1] and 
                  rsi_current_aligned < 50 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: fast MA crosses below slow MA
            if ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: fast MA crosses above slow MA
            if ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_MA_Crossover_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0