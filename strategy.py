#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter (EMA50) and volume confirmation
# Long when price touches Camarilla L3 level AND price > daily EMA50 AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 level AND price < daily EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back through the Camarilla H4/L4 levels (strong reversal)
# This captures mean-reversion bounces in established trends while avoiding counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's range
    # H4 = close + 1.5 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first value to NaN since we don't have previous day data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1D data to 4H timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price touches L3 + above daily EMA50 + volume confirmation
            if (price <= L3_aligned[i] * 1.002 and price >= L3_aligned[i] * 0.998 and 
                price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 + below daily EMA50 + volume confirmation
            elif (price >= H3_aligned[i] * 0.998 and price <= H3_aligned[i] * 1.002 and 
                  price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back through H4 (strong reversal)
            if price >= H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back through L4 (strong reversal)
            if price <= L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0