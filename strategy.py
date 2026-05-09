#!/usr/bin/env python3
# Hypothesis: 1-hour Camarilla pivot points with 4-hour EMA trend filter and volume confirmation
# Long when price breaks above Camarilla R3, 4h EMA(50) rising, and volume >1.5x 20-period average
# Short when price breaks below Camarilla S3, 4h EMA(50) falling, and volume >1.5x 20-period average
# Exit when price crosses Camarilla H4 (pivot) or trend reverses
# Position size: 0.20 to limit drawdown. Target: 15-30 trades/year (60-120 over 4 years).
# Works in bull (breakouts) and bear (mean reversion at extremes) via trend filter.

name = "1h_Camarilla_R3S3_4hEMA50_Volume"
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
    
    # Calculate Camarilla pivot points (based on previous period's OHLC)
    # For 1h chart, use previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    h4 = pivot + (range_hl * 1.1 / 4)  # H4 is midpoint for exit
    l4 = pivot - (range_hl * 1.1 / 4)  # L4 is midpoint for exit
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close']
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_prev = np.roll(ema_50_4h, 1)
    ema_50_4h_prev[0] = ema_50_4h[0]
    ema_rising = ema_50_4h > ema_50_4h_prev
    ema_falling = ema_50_4h < ema_50_4h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 4h EMA rising + volume spike
            if (close[i] > r3[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price < Camarilla S3 + 4h EMA falling + volume spike
            elif (close[i] < s3[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below H4 (Camarilla midpoint) OR trend turns down
            if (close[i] < h4[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above L4 (Camarilla midpoint) OR trend turns up
            if (close[i] > l4[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals