# 12h_1d_Camarilla_Pivot_Breakout_Volume
# Hypothesis: 12-hour Camarilla pivot breakout with 1-day EMA trend filter and volume confirmation
# Long when price closes above H3 AND price > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price closes below L3 AND price < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Camarilla H3/L3 levels
# Camarilla pivots are effective reversal points in ranging markets, EMA filters for trend alignment, volume confirms breakouts
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d period
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 50 for EMA)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: close above H3 + above 1d EMA50 + volume confirmation
            if (price > camarilla_h3_aligned[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: close below L3 + below 1d EMA50 + volume confirmation
            elif (price < camarilla_l3_aligned[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes back inside H3/L3 (below H3)
            if price < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes back inside H3/L3 (above L3)
            if price > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0