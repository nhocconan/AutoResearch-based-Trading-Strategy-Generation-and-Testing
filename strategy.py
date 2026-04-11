#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1d Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide strong support/resistance. Long at L3/S3 bounce with weekly uptrend, short at H3/H4 rejection with weekly downtrend. Volume spike confirms institutional interest. Designed for low frequency (10-25 trades/year) to minimize fee drag in all market regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Calculate daily Camarilla pivot levels
        # Based on previous day's OHLC
        if i < 1:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_range = prev_high - prev_low
        
        if prev_range <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla levels
        # H4 = close + 1.5 * range * 1.1
        # H3 = close + 1.25 * range * 1.1
        # L3 = close - 1.1 * range * 1.1
        # L4 = close - 2 * range * 1.1
        h4 = prev_close + 1.5 * prev_range * 1.1
        h3 = prev_close + 1.25 * prev_range * 1.1
        l3 = prev_close - 1.1 * prev_range * 1.1
        l4 = prev_close - 2.0 * prev_range * 1.1
        
        # Volume spike: current volume > 1.8x 20-day average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (vol_ma_20 * 1.8)
        else:
            volume_spike = False
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry logic: Camarilla bounce/rejection + volume + trend
        long_signal = (close[i] <= l3 * 1.002 and close[i] >= l4 * 0.998 and 
                      volume_spike and uptrend and position != 1)
        short_signal = (close[i] >= h3 * 0.998 and close[i] <= h4 * 1.002 and 
                       volume_spike and downtrend and position != -1)
        
        if long_signal:
            position = 1
            signals[i] = 0.25
        elif short_signal:
            position = -1
            signals[i] = -0.25
        # Exit: price moves to opposite side of pivot point
        elif position == 1 and close[i] >= (prev_high + prev_low) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= (prev_high + prev_low) / 2:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals