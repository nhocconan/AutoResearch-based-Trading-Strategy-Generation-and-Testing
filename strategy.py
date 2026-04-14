#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Camarilla pivot with weekly trend filter (EMA10) and volume confirmation
# Long when price touches Camarilla L3 support AND price > weekly EMA10 AND volume > 1.5x 20-day average
# Short when price touches Camarilla H3 resistance AND price < weekly EMA10 AND volume > 1.5x 20-day average
# Exit when price crosses Camarilla H4 (long exit) or L4 (short exit)
# This captures mean-reversion in weekly trends with volume confirmation, avoiding counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla pivot levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    H4 = pivot + (range_val * 1.5 / 2)
    H3 = pivot + (range_val * 1.25 / 2)
    L3 = pivot - (range_val * 1.25 / 2)
    L4 = pivot - (range_val * 1.5 / 2)
    
    # Calculate weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate volume average for confirmation (20-day)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(H4[i]) or np.isnan(L4[i]) or
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: touch L3 support AND above weekly EMA10 AND volume confirmation
            if (low[i] <= L3[i] and price > ema10_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: touch H3 resistance AND below weekly EMA10 AND volume confirmation
            elif (high[i] >= H3[i] and price < ema10_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above H4 resistance
            if high[i] >= H4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below L4 support
            if low[i] <= L4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Camarilla_1wEMA10_Volume"
timeframe = "1d"
leverage = 1.0