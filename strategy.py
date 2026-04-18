#!/usr/bin/env python3
"""
12h_1w_CCI_Extreme_Reversion_With_Volume
Hypothesis: Weekly CCI extremes (>200 or <-200) indicate overextended moves likely to revert, with volume confirmation on 12h.
In bear markets (2022), extreme readings often precede mean-reversion bounces; in bull markets, they catch overextended pullbacks.
Uses 12h close > 20-period EMA for long filter and < 20-period EMA for short filter to align with intermediate trend.
Targets 15-25 trades/year to minimize fee drag while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly CCI for extreme readings (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate CCI(20): (Typical Price - MA) / (0.015 * Mean Deviation)
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    cci_period = 20
    ma_tp = pd.Series(tp_1w).rolling(window=cci_period, min_periods=cci_period).mean()
    mad = pd.Series(tp_1w).rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    cci_20 = (tp_1w - ma_tp.values) / (0.015 * mad.values)
    cci_20 = np.nan_to_num(cci_20, nan=0.0)
    
    cci_20_aligned = align_htf_to_ltf(prices, df_1w, cci_20)
    
    # 12h EMA20 for trend alignment
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(cci_20_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        cci = cci_20_aligned[i]
        ema20 = ema_20[i]
        vol_spike = volume_spike[i]
        price = close[i]
        
        if position == 0:
            # Long: CCI < -200 (extreme oversold) with volume spike and price above EMA20
            if cci < -200 and vol_spike and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: CCI > 200 (extreme overbought) with volume spike and price below EMA20
            elif cci > 200 and vol_spike and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: CCI returns above -50 (reversion complete) or trend weakens
            if cci > -50 or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: CCI returns below 50 (reversion complete) or trend weakens
            if cci < 50 or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_CCI_Extreme_Reversion_With_Volume"
timeframe = "12h"
leverage = 1.0