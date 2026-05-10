# 4h_VolumeSpike_1dTrend_Pullback
# Hypothesis: On 4h timeframe, buy when volume spikes (>2x 20-period MA) and price pulls back to 1d EMA34 in uptrend; sell when volume spikes and price rallies to 1d EMA34 in downtrend.
# This captures mean reversion within the higher timeframe trend, reducing whipsaw. Volume spikes indicate institutional interest. Works in bull/bear via 1d trend filter.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_VolumeSpike_1dTrend_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume 20-period MA for spike detection
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 4h data for price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 2x 1d 20-period MA
        volume_spike = volume[i] > vol_ma20_1d_aligned[i] * 2.0
        
        # Trend filter: price vs 1d EMA34
        price_near_ema = abs(close[i] - ema34_1d_aligned[i]) / ema34_1d_aligned[i] < 0.015  # within 1.5%
        
        if position == 0:
            # Long: volume spike, price near 1d EMA34, and above EMA (uptrend)
            if volume_spike and price_near_ema and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: volume spike, price near 1d EMA34, and below EMA (downtrend)
            elif volume_spike and price_near_ema and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves 1.5% away from EMA or volume spike in opposite direction
            if abs(close[i] - ema34_1d_aligned[i]) / ema34_1d_aligned[i] > 0.015 or \
               (volume[i] > vol_ma20_1d_aligned[i] * 2.0 and close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves 1.5% away from EMA or volume spike in opposite direction
            if abs(close[i] - ema34_1d_aligned[i]) / ema34_1d_aligned[i] > 0.015 or \
               (volume[i] > vol_ma20_1d_aligned[i] * 2.0 and close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals