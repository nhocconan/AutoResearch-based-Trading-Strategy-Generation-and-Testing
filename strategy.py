#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels with daily trend filter and volume confirmation
# Long when price breaks above Camarilla R4 (bullish breakout) AND price > daily EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla S4 (bearish breakdown) AND price < daily EMA200 AND volume > 1.5x 20-period average
# Exit when price crosses back to the Camarilla Pivot point (mean reversion to equilibrium)
# Camarilla levels provide precise intraday support/resistance based on prior day's range
# EMA200 filter ensures trading with the higher timeframe trend
# Volume confirmation avoids false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation and EMA200
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe (available after daily candle closes)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate daily EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (200 for EMA + buffer)
    start = 220
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Camarilla R4 + above daily EMA200 + volume confirmation
            if (price > r4_aligned[i] and price > ema200_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Camarilla S4 + below daily EMA200 + volume confirmation
            elif (price < s4_aligned[i] and price < ema200_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla Pivot (mean reversion)
            if price <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla Pivot (mean reversion)
            if price >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0