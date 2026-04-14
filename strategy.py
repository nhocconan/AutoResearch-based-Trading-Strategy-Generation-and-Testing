#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla L3 breakout with weekly trend filter and volume confirmation
# Long when price breaks above Camarilla L3 level AND weekly EMA(50) is rising AND volume > 1.5x average
# Short when price breaks below Camarilla L3 level AND weekly EMA(50) is falling AND volume > 1.5x average
# Exit when price crosses back through Camarilla L4 (mean reversion) or opposite breakout occurs
# Camarilla levels provide precise support/resistance; weekly EMA ensures higher timeframe trend alignment; volume confirms institutional interest
# Designed to work in both bull and bear markets by following the dominant trend on weekly timeframe
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels on 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    L3 = pivot + (range_1d * 1.1 / 4)
    L4 = pivot + (range_1d * 1.1 / 2)
    H3 = pivot - (range_1d * 1.1 / 4)
    H4 = pivot - (range_1d * 1.1 / 2)
    
    # Calculate EMA on weekly (50-period) for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d Camarilla levels and weekly EMA to 12h timeframe
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(L3_12h[i]) or 
            np.isnan(L4_12h[i]) or 
            np.isnan(H3_12h[i]) or 
            np.isnan(H4_12h[i]) or 
            np.isnan(ema_50_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 12h timeframe
        ema_val = ema_50_12h[i]
        ema_prev = ema_50_12h[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above L3 level AND weekly EMA rising AND volume confirmation
            if (high_val > L3_12h[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below H3 level AND weekly EMA falling AND volume confirmation
            elif (low_val < H3_12h[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below L4 level OR opposite breakout
            if (close_val < L4_12h[i] or 
                low_val < H3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above H4 level OR opposite breakout
            if (close_val > H4_12h[i] or 
                high_val > L3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_L3_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0