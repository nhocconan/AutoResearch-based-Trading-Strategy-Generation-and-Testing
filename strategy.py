#!/usr/bin/env python3
"""
6h_HighLowBand_12hTrend_VolumeBreakout
Hypothesis: 6h price breaks above/below dynamic high-low band (based on 12h ATR) with 12h EMA trend filter and volume confirmation.
In bull markets: buy breakouts above upper band with uptrend.
In bear markets: sell breakouts below lower band with downtrend.
The adaptive band adjusts to volatility, reducing whipsaws in ranging markets. Volume confirmation ensures breakout validity.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h ATR(14) for dynamic band width
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # ATR(14) with min_periods
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Dynamic band: upper/lower = 12h close ± 2.0 * ATR(14)
    upper_band = close_12h + 2.0 * atr_14_12h
    lower_band = close_12h - 2.0 * atr_14_12h
    
    # Align bands to 6h timeframe (completed 12h bar only)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h ATR(14), EMA50, and volume MA
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band + 12h uptrend + volume spike
            long_setup = (close[i] > upper_band_aligned[i]) and (close[i] > ema_50_12h_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower band + 12h downtrend + volume spike
            short_setup = (close[i] < lower_band_aligned[i]) and (close[i] < ema_50_12h_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below lower band OR 12h trend turns down
            if (close[i] < lower_band_aligned[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper band OR 12h trend turns up
            if (close[i] > upper_band_aligned[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_HighLowBand_12hTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0