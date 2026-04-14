#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot bounce with 1-day trend filter (EMA50) and volume confirmation
# Long when price touches Camarilla L3 support AND price > daily EMA50 AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 resistance AND price < daily EMA50 AND volume > 1.5x 20-period average
# Exit when price reaches opposite Camarilla level (L3 for shorts, H3 for longs) or reverses at opposite H3/L3
# Camarilla levels provide precise intraday support/resistance, effective in ranging markets
# EMA50 filter ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation filters weak breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for 12h using previous 12h OHLC
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use H3/L3 as entry levels and H4/L4 as stop levels
    
    # Calculate rolling window for previous 12h bar (need high, low, close of previous bar)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN since no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    diff = prev_high - prev_low
    H3 = prev_close + 1.1 * diff
    L3 = prev_close - 1.1 * diff
    H4 = prev_close + 1.5 * diff
    L4 = prev_close - 1.5 * diff
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price touches L3 support AND price > daily EMA50 AND volume confirmation
            if (price <= L3[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 resistance AND price < daily EMA50 AND volume confirmation
            elif (price >= H3[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 resistance (opposite level) or shows rejection at H4
            if price >= H3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 support (opposite level) or shows rejection at L4
            if price <= L3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_L3H3_EMA50_Volume"
timeframe = "12h"
leverage = 1.0