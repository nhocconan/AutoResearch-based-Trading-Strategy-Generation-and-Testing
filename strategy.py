#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot bounce with 1-day trend filter (EMA200) and volume confirmation
# Long when price touches Camarilla L3 level AND price > daily EMA200 AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 level AND price < daily EMA200 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Camarilla range (H3-L3)
# Uses Camarilla levels from daily timeframe for stronger support/resistance
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and EMA200
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on daily data
    # Camarilla: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula: H3 = Close + (High - Low) * 1.1/4, L3 = Close - (High - Low) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Calculate daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (200 for EMA200 + buffer)
    start = 210
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: touch Camarilla L3 + above daily EMA200 + volume confirmation
            if (price <= camarilla_l3_aligned[i] and price > ema200_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: touch Camarilla H3 + below daily EMA200 + volume confirmation
            elif (price >= camarilla_h3_aligned[i] and price < ema200_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price rises back above Camarilla H3 (upper range)
            if price >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price falls back below Camarilla L3 (lower range)
            if price <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_EMA200_Volume"
timeframe = "12h"
leverage = 1.0