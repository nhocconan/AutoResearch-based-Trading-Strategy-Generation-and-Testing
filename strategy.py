#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot level breakout with 1-day trend filter and volume confirmation
# Long when price breaks above H3 level AND price > daily EMA50 AND volume > 1.3x 20-period average
# Short when price breaks below L3 level AND price < daily EMA50 AND volume > 1.3x 20-period average
# Exit when price crosses back inside the H3/L3 range (between H3 and L3)
# Camarilla levels derived from previous 1-day OHLC; effective in ranging and trending markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    # We use H3/L3 as primary breakout levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero or invalid calculations
    hl_range = prev_high - prev_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # prevent zero range
    
    H3 = prev_close + hl_range * 1.1 / 4
    L3 = prev_close - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels for current day)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need at least 1 day of history)
    start = 1
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.3
        
        if position == 0:
            # Long setup: breakout above H3 + above daily EMA50 + volume confirmation
            if (price > H3_aligned[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L3 + below daily EMA50 + volume confirmation
            elif (price < L3_aligned[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L3 (return to neutral zone)
            if price < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H3 (return to neutral zone)
            if price > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0