#!/usr/bin/env python3
"""
12h Daily Close Position + Volume Confirmation + ATR Filter (Revised)
Hypothesis: Daily close above/below key levels with volume and momentum provides directional bias for 12h.
In bull markets, price tends to close above prior day's high; in bear markets, below prior day's low.
Volume confirms conviction. ATR filter avoids choppy markets. Designed for 15-30 trades/year.
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
    
    # Get daily data for prior day's levels (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Prior day's high and low
    prev_daily_high = df_d['high'].shift(1).values  # shift(1) for prior day
    prev_daily_low = df_d['low'].shift(1).values
    
    # Align daily levels to 12h timeframe (already delayed by shift)
    high_aligned = align_htf_to_ltf(prices, df_d, prev_daily_high)
    low_aligned = align_htf_to_ltf(prices, df_d, prev_daily_low)
    
    # ATR for volatility filter (daily ATR)
    tr1 = df_d['high'] - df_d['low']
    tr2 = abs(df_d['high'] - df_d['close'].shift(1))
    tr3 = abs(df_d['low'] - df_d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_d, atr_d)
    
    # Volume spike detection (2x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(high_aligned[i]) or 
            np.isnan(low_aligned[i]) or
            np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_high = high_aligned[i]
        prev_low = low_aligned[i]
        atr = atr_aligned[i]
        
        # Avoid choppy markets: require ATR > 0.5 * price (adjust as needed)
        if atr < 0.005 * price:  # too low volatility
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close above prior day's high with volume spike
            if price > prev_high and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below prior day's low with volume spike
            elif price < prev_low and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to prior day's low or ATR-based trailing stop
            if price <= prev_low or price < (high_aligned[i] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to prior day's high or ATR-based trailing stop
            if price >= prev_high or price > (low_aligned[i] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DailyClose_Position_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0