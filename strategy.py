#!/usr/bin/env python3
"""
4h CCI(20) Mean Reversion with Volume Spike Filter
Long: CCI < -100 + volume > 1.5 x 4h volume MA(20)
Short: CCI > +100 + volume > 1.5 x 4h volume MA(20)
Exit: CCI crosses back above -50 (long) or below +50 (short)
Uses mean reversion in oversold/overbought conditions with volume confirmation.
Target: 30-50 trades/year per symbol (120-200 total over 4 years)
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
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    
    # Avoid division by zero
    cci = np.zeros_like(typical_price, dtype=float)
    mask = tp_std != 0
    cci[mask] = (typical_price[mask] - tp_ma[mask]) / (0.015 * tp_std[mask])
    
    # 4h volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # wait for CCI and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(cci[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        cci_val = cci[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: CCI < -100 (oversold) + volume spike
            if cci_val < -100.0 and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: CCI > +100 (overbought) + volume spike
            elif cci_val > 100.0 and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses back above -50
            if cci_val > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses back below +50
            if cci_val < 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CCI20_MeanReversion_VolumeSpike"
timeframe = "4h"
leverage = 1.0