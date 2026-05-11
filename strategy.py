# [Experiment #153782] 12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
# Hypothesis: On 12h timeframe, trade Camarilla pivot breakouts (R1/S1) with 1d EMA trend filter and volume spike confirmation.
# Works in bull markets (breakouts continue trend) and bear markets (mean reversion at extremes with volume confirmation).
# Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # === 1D Data for Trend and Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 20)  # EMA34 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in uptrend (close > EMA34)
            if close[i] > r1_1d_aligned[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in downtrend (close < EMA34)
            elif close[i] < s1_1d_aligned[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (mean reversion) or trend breaks down
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 (mean reversion) or trend breaks up
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals