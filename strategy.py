#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels from 1d for entry/exit signals
# 1d EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 1d trend
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_val = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d_val) / 3
    range_1d = high_1d - low_1d
    # Camarilla levels: H4 = close + 1.1/2 * range, L4 = close - 1.1/2 * range
    h4 = close_1d_val + 1.1 * range_1d / 2
    l4 = close_1d_val - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)  # re-align EMA
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above H4 with volume filter and above 1d EMA
            if price > h4_aligned[i] and vol > 1.5 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L4 with volume filter and below 1d EMA
            elif price < l4_aligned[i] and vol > 1.5 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L4 (reversal) or below 1d EMA
            if price < l4_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H4 (reversal) or above 1d EMA
            if price > h4_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0