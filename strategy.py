#!/usr/bin/env python3
# 4H_Relative_Volume_Pullback_to_EMA200
# Hypothesis: Combines relative volume spike with pullback to 200-period EMA on 4h chart.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
# Uses volume confirmation to avoid false breakouts and EMA200 as dynamic support/resistance.
# Target: 20-40 trades per year per symbol with clear entry/exit rules.

name = "4H_Relative_Volume_Pullback_to_EMA200"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 200-period EMA for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Relative volume: current volume / average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rel_volume = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to be valid
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if np.isnan(ema200[i]) or np.isnan(rel_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: relative volume > 1.5 (volume spike)
        volume_filter = rel_volume[i] > 1.5
        
        if position == 0:
            # Long: Price pulls back to EMA200 from above + volume spike
            if (close[i] <= ema200[i] * 1.01 and  # Within 1% above EMA200
                close[i-1] > ema200[i-1] and      # Was above EMA200 previous bar
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price pulls back to EMA200 from below + volume spike
            elif (close[i] >= ema200[i] * 0.99 and  # Within 1% below EMA200
                  close[i-1] < ema200[i-1] and      # Was below EMA200 previous bar
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit when price moves 1.5% away from EMA200 in favor of position
            if position == 1 and close[i] >= ema200[i] * 1.015:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= ema200[i] * 0.985:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals