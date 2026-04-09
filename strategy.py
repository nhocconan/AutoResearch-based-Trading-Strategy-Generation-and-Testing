#!/usr/bin/env python3
# 6h_volatility_breakout_volume_v2
# Hypothesis: 6h strategy using 1d ATR-based volatility breakout with volume confirmation.
# In high volatility regimes: price breaks above/below ATR(20) bands from open + volume spike → enter direction of breakout.
# In low volatility: wait for expansion. Uses discrete sizing (±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via volatility regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volatility_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(20) on 1d
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(high_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1.abs(), tr2.abs(), tr3.abs()], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate breakout levels: open ± 0.5 * ATR(1d)
        upper_break = open_[i] + 0.5 * atr_1d_aligned[i]
        lower_break = open_[i] - 0.5 * atr_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below open (mean reversion)
            if close[i] < open_[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above open (mean reversion)
            if close[i] > open_[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and volatility expansion
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            volatility_expanding = atr_1d_aligned[i] > pd.Series(atr_1d_aligned).rolling(window=10, min_periods=10).mean().iloc[i] if i >= 10 else False
            
            if volume_confirmed and volatility_expanding:
                # Long: price breaks above upper band
                if close[i] > upper_break:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower band
                elif close[i] < lower_break:
                    position = -1
                    signals[i] = -0.25
    
    return signals