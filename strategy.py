#!/usr/bin/env python3
# 6h_1d_cci_volume_v1
# Strategy: 6h CCI extreme reversal with daily volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI > +100 indicates overbought, < -100 indicates oversold.
# In ranging markets (common in BTC/ETH 2025), extreme CCI readings often precede mean reversion.
# Daily volume > 1.5x 20-period average confirms participation.
# Works in bull markets via shorting overextended rallies and bear markets via buying capitulation.
# Low trade frequency expected (~15-30/year) due to strict CCI extremes + volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h CCI (20-period)
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    cci_values = cci.values
    
    # 1d volume average (20-period)
    vol_avg_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(cci_values[i]) or np.isnan(vol_avg_20_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Daily volume confirmation: current 6h bar volume > 1.5x 20-period daily average
        # Note: Comparing 6h volume to daily average requires scaling
        # Approximate: 6h volume should be > (1.5 * daily_avg / 4) since 4x6h = 1d
        vol_confirm = volume[i] > 1.5 * vol_avg_20_aligned[i] / 4.0
        
        # CCI extreme conditions
        cci_overbought = cci_values[i] > 100
        cci_oversold = cci_values[i] < -100
        
        # Entry conditions
        # Long: CCI oversold AND volume confirmation
        if cci_oversold and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI overbought AND volume confirmation
        elif cci_overbought and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone (-50 to 50)
        elif position == 1 and cci_values[i] < 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_values[i] > -50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals