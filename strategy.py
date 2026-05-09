#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot (S1/R1) mean reversion with 12h EMA trend filter and volume confirmation
# Long when: price < S1 AND 12h EMA(50) rising AND volume spike (>1.5x 20-period average)
# Short when: price > R1 AND 12h EMA(50) falling AND volume spike
# Exit when: price crosses pivot point (PP) OR trend reverses
# Position size: 0.25 to limit drawdown. Target: 25-50 trades/year.
# Designed to work in both bull (mean-reversion at support) and bear (mean-reversion at resistance) markets.

name = "4h_Camarilla_S1R1_12hEMA_Volume"
timeframe = "4h"
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
    
    # Calculate Camarilla pivot levels (based on previous bar's OHLC)
    # Using previous bar to avoid look-ahead
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels
    S1 = close_prev - (range_val * 1.0 / 12.0)
    S2 = close_prev - (range_val * 2.0 / 12.0)
    S3 = close_prev - (range_val * 3.0 / 12.0)
    S4 = close_prev - (range_val * 4.0 / 12.0)
    R1 = close_prev + (range_val * 1.0 / 12.0)
    R2 = close_prev + (range_val * 2.0 / 12.0)
    R3 = close_prev + (range_val * 3.0 / 12.0)
    R4 = close_prev + (range_val * 4.0 / 12.0)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close']
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_rising = ema_50_12h > ema_50_12h_prev
    ema_falling = ema_50_12h < ema_50_12h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(S1[i]) or np.isnan(R1[i]) or np.isnan(pivot[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price < S1 AND 12h EMA rising AND volume spike
            if (close[i] < S1[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price > R1 AND 12h EMA falling AND volume spike
            elif (close[i] > R1[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above pivot point OR trend turns down
            if (close[i] > pivot[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below pivot point OR trend turns up
            if (close[i] < pivot[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals