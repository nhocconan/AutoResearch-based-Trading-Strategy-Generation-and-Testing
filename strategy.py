#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and RSI Filter
Targets reversals at key Camarilla levels (R1/S1) with volume confirmation and RSI momentum filter.
Designed for low trade frequency (target: 20-50 trades/year) with strong edge in ranging markets.
Uses 1d Camarilla levels for structure and 1h RSI for momentum confirmation.
Works in both bull and bear markets by fading extremes at key pivot levels.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for current day (based on previous day)
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's OHLC to calculate today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    rang = prev_high - prev_low
    camarilla_r1 = prev_close + rang * 1.1 / 12
    camarilla_s1 = prev_close - rang * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1h data for RSI filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi)
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long: price touches S1 with volume spike and RSI oversold (<30)
            if (abs(price - s1) < 0.001 * s1 and  # Within 0.1% of S1
                volume_spike[i] and 
                rsi_val < 30):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price touches R1 with volume spike and RSI overbought (>70)
            elif (abs(price - r1) < 0.001 * r1 and  # Within 0.1% of R1
                  volume_spike[i] and 
                  rsi_val > 70):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price moves back above midpoint or RSI overbought
            midpoint = (r1 + s1) / 2
            if price > midpoint or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price moves back below midpoint or RSI oversold
            midpoint = (r1 + s1) / 2
            if price < midpoint or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_Volume_RSI"
timeframe = "4h"
leverage = 1.0