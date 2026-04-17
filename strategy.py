#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d ATR Volatility Regime with Volume Confirmation.
Long when Williams %R < -80 (oversold) and ATR(14) > ATR(50) (high volatility regime) with volume > 1.5x average.
Short when Williams %R > -20 (overbought) and ATR(14) > ATR(50) with volume > 1.5x average.
Exit when Williams %R returns to -50 (mean reversion) or volatility regime ends (ATR(14) <= ATR(50)).
Uses 6h for price/W%R/volume, 1d for ATR regime filter.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R catches reversals in high volatility, ATR filter ensures we trade only when momentum is sustainable.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime
    def calculate_atr(high, low, close, period):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        if len(close) >= period:
            atr[period-1] = np.mean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr50_1d = calculate_atr(high_1d, low_1d, close_1d, 50)
    
    # Volatility regime: ATR(14) > ATR(50) = expanding volatility (good for momentum)
    vol_regime = atr14_1d > atr50_1d
    
    # Align 1d indicators to 6h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Calculate 6h Williams %R(14)
    def calculate_williams_r(high, low, close, period):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            start_idx = max(0, i - period + 1)
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        
        wr = np.zeros_like(close)
        for i in range(len(close)):
            if highest_high[i] != lowest_low[i]:
                wr[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
            else:
                wr[i] = -50  # neutral when no range
        return wr
    
    wr_6h = calculate_williams_r(high, low, close, 14)
    
    # Calculate 6h volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_6h[i]) or 
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = wr_6h[i]
        vol_reg = vol_regime_aligned[i] > 0.5  # convert back to boolean
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in high volatility regime with volume spike
            if wr < -80 and vol_reg and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in high volatility regime with volume spike
            elif wr > -20 and vol_reg and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or volatility regime ends
            if wr >= -50 or not vol_reg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or volatility regime ends
            if wr <= -50 or not vol_reg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dATR_VolatilityRegime_Volume"
timeframe = "6h"
leverage = 1.0