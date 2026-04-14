#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H4 level AND 1d EMA(50) is rising AND volume > 1.5x average
# Short when price breaks below 1d Camarilla L4 level AND 1d EMA(50) is falling AND volume > 1.5x average
# Exit when price crosses back through 1d Camarilla midpoint (H4+L4)/2 or opposite breakout occurs
# Camarilla levels provide statistically significant intraday support/resistance
# 1d EMA ensures higher timeframe trend alignment; volume confirms institutional interest
# Designed to work in both bull and bear markets by following the dominant trend on 1d timeframe
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and maximize edge

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d OHLC
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_H4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_L4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_mid = (camarilla_H4 + camarilla_L4) / 2
    
    # Calculate EMA on 1d (50-period) for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d indicators to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4.values)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4.values)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i])):
            signals[i] = 0.0
            continue
        
        camarilla_H4_val = camarilla_H4_aligned[i]
        camarilla_L4_val = camarilla_L4_aligned[i]
        camarilla_mid_val = camarilla_mid_aligned[i]
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Camarilla H4 AND 1d EMA rising AND volume confirmation
            if (high_val > camarilla_H4_val and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Camarilla L4 AND 1d EMA falling AND volume confirmation
            elif (low_val < camarilla_L4_val and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Camarilla midpoint OR opposite breakout
            if (close_val < camarilla_mid_val or 
                low_val < camarilla_L4_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Camarilla midpoint OR opposite breakout
            if (close_val > camarilla_mid_val or 
                high_val > camarilla_H4_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0