#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_extreme_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily CCI (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp = (high_1d + low_1d + close_1d) / 3
    
    # Moving average of typical price
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # CCI calculation
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # Volume confirmation: 12h volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align daily CCI to 12h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 2.0 * vol_ma_50[i]
        
        # CCI extreme conditions
        cci_overbought = cci_aligned[i] > 100
        cci_oversold = cci_aligned[i] < -100
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: CCI crosses above -100 from oversold with volume confirmation
        if i > 20:
            cci_prev = cci_aligned[i-1]
            cci_cross_up = (cci_prev <= -100) and (cci_aligned[i] > -100)
            if cci_cross_up and vol_confirm:
                enter_long = True
        
        # Short: CCI crosses below 100 from overbought with volume confirmation
        if i > 20:
            cci_prev = cci_aligned[i-1]
            cci_cross_down = (cci_prev >= 100) and (cci_aligned[i] < 100)
            if cci_cross_down and vol_confirm:
                enter_short = True
        
        # Exit conditions: return to zero line
        exit_long = cci_aligned[i] < 0
        exit_short = cci_aligned[i] > 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h CCI extreme reversal strategy with volume confirmation.
# Enters long when daily CCI crosses above -100 from oversold territory with volume > 2x 50-period average.
# Enters short when daily CCI crosses below 100 from overbought territory with volume > 2x 50-period average.
# Exits when CCI returns to zero line.
# Uses extreme CCI levels (>100/< -100) to filter noise and volume confirmation to ensure conviction.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by capturing reversals at extremes.
# 12h timeframe reduces noise and daily CCI provides institutional-grade momentum reading.