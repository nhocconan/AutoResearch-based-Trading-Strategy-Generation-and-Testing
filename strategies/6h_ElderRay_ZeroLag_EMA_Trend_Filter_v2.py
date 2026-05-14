#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_Filter_v2
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ZeroLag EMA trend filter and volume confirmation.
Enters long when Bull Power > 0, price above ZeroLag EMA(50), and volume spike.
Enters short when Bear Power < 0, price below ZeroLag EMA(50), and volume spike.
Exits when Elder Ray reverses or price crosses ZeroLag EMA.
Uses 6h primary timeframe to target 12-37 trades/year (50-150 total over 4 years).
ZeroLag EMA reduces lag while maintaining trend-following capability.
Works in bull/bear markets by aligning with 6h trend to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def zerolag_ema(data, period):
    """Calculate ZeroLag EMA"""
    ema1 = pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    return (2 * ema1 - ema2).values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray and ZeroLag EMA calculation (self-referential)
    df_6h = prices.copy()  # Primary timeframe is 6h
    
    # Calculate EMA13 and EMA34 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA34
    bull_power = high - ema13
    bear_power = low - ema34
    
    # ZeroLag EMA(50) for trend filter
    zl_ema = zerolag_ema(close, 50)
    
    # Volume confirmation: volume > 1.8x 30-period MA
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 30 for volume MA)
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or np.isnan(zl_ema[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Bull Power > 0, price above ZeroLag EMA, volume spike
            if (bull_power[i] > 0 and 
                close[i] > zl_ema[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, price below ZeroLag EMA, volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < zl_ema[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR price crosses below ZeroLag EMA
            if (bull_power[i] <= 0 or close[i] < zl_ema[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR price crosses above ZeroLag EMA
            if (bear_power[i] >= 0 or close[i] > zl_ema[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_Filter_v2"
timeframe = "6h"
leverage = 1.0