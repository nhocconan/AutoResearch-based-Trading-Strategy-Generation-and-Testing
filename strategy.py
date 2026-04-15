# 6h Volume-Weighted Gap Reversal
# Hypothesis: On 6B timeframe, significant volume spikes (>2x median volume) combined with
# price gaps (>0.5% from prior close) often reverse in mean-reverting fashion.
# Long when: gap down >0.5% + volume spike >2x median + price near 6h low
# Short when: gap up >0.5% + volume spike >2x median + price near 6h high
# Works in both bull/bear markets as volatility spikes create reversals regardless of trend.
# Target: 20-40 trades/year (80-160 total) with low turnover to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detection: current volume > 2x 50-period median
    vol_median = pd.Series(volume).rolling(window=50, min_periods=20).median().values
    vol_spike = volume > (2.0 * vol_median)
    
    # Gap detection: abs((open - prev_close) / prev_close) > 0.005 (0.5%)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    gap_pct = (prices['open'].values - prev_close) / prev_close
    gap_up = gap_pct > 0.005
    gap_down = gap_pct < -0.005
    
    # Price position within 6h range: near high/low for fade signals
    hl_range = high - low
    near_high = (high - close) < (0.1 * hl_range)  # within 10% of high
    near_low = (close - low) < (0.1 * hl_range)    # within 10% of low
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(50, n):
        # Skip if volume median not ready
        if np.isnan(vol_median[i]):
            continue
            
        # Long setup: gap down + volume spike + near low
        if gap_down[i] and vol_spike[i] and near_low[i] and position <= 0:
            position = 1
            signals[i] = position_size
        # Short setup: gap up + volume spike + near high
        elif gap_up[i] and vol_spike[i] and near_high[i] and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: reverse signal or loss of momentum
        elif position == 1 and (gap_up[i] or not vol_spike[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (gap_down[i] or not vol_spike[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_VolumeGap_Reversal"
timeframe = "6h"
leverage = 1.0