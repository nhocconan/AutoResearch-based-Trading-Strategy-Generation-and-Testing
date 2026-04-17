#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d Volatility Regime Filter.
Long when Williams %R crosses above -80 from below in low volatility regime (1d ATR ratio < 1.2).
Short when Williams %R crosses below -20 from above in low volatility regime.
Exit when Williams %R crosses opposite threshold (-20 for long exit, -80 for short exit) or volatility spikes (ATR ratio > 1.5).
Uses 1d for ATR-based volatility regime, 6h for Williams %R oscillator.
Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by fading extreme momentum during low volatility periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volatility regime (ATR)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # Calculate 1d ATR ratio (current ATR / 50-period SMA of ATR) for volatility regime
    atr_14 = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    
    # Calculate 6h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    # Align 1d indicators
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(wr_14[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = atr_ratio_aligned[i]
        wr = wr_14[i]
        wr_prev = wr_14[i-1]
        
        # Low volatility regime: ATR ratio < 1.2 (below average volatility)
        # High volatility regime: ATR ratio > 1.5 (above average volatility)
        is_low_vol = vol_ratio < 1.2
        is_high_vol = vol_ratio > 1.5
        
        # Williams %R signals
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        wr_cross_up_80 = wr_prev <= -80 and wr > -80  # crosses above -80
        wr_cross_down_20 = wr_prev >= -20 and wr < -20  # crosses below -20
        wr_cross_down_80 = wr_prev >= -80 and wr < -80  # crosses below -80 (exit long)
        wr_cross_up_20 = wr_prev <= -20 and wr > -20   # crosses above -20 (exit short)
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below in low volatility
            if wr_cross_up_80 and is_low_vol:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above in low volatility
            elif wr_cross_down_20 and is_low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -80 OR volatility spikes
            if wr_cross_down_80 or is_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -20 OR volatility spikes
            if wr_cross_up_20 or is_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dATR_VolatilityRegime"
timeframe = "6h"
leverage = 1.0