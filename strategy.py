#/usr/bin/env python3
# 6h_Fisher_Transform_1dTrend_Reversal
# Hypothesis: Ehlers Fisher Transform on 6h for mean reversion signals, filtered by 1d trend direction.
# Fisher Transform identifies turning points in price distributions; trades counter-trend in strong trends.
# In bull markets, takes short signals during pullbacks in uptrend; in bear markets, takes long signals during bounces in downtrend.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in weak trends.
# Targets 15-25 trades/year to minimize fee drag. Works in all regimes by trading reversals with trend filter.

name = "6h_Fisher_Transform_1dTrend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Ehlers Fisher Transform on 6h prices (period=10)
    # Price normalization to [-1, 1] range
    hlc = (high + low + close) / 3.0
    max_hlc = pd.Series(hlc).rolling(window=10, min_periods=10).max().values
    min_hlc = pd.Series(hlc).rolling(window=10, min_periods=10).min().values
    range_hlc = max_hlc - min_hlc
    
    # Avoid division by zero
    value1 = np.where(range_hlc > 0, 2 * ((hlc - min_hlc) / range_hlc - 0.5), 0)
    # Limit value1 to [-0.999, 0.999] for Fisher transform
    value1 = np.clip(value1, -0.999, 0.999)
    
    # Fisher Transform
    fish = np.zeros_like(hlc)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value1[i]) / (1 - value1[i])) + 0.5 * fish[i-1]
    
    # Smooth Fisher Transform (3-period EMA)
    fish_smoothed = pd.Series(fish).ewm(span=3, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(fish_smoothed[i]) or np.isnan(max_hlc[i]) or np.isnan(min_hlc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 in uptrend (mean reversion long in uptrend pullback)
            if (fish_smoothed[i] > -1.5 and fish_smoothed[i-1] <= -1.5 and
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 in downtrend (mean reversion short in downtrend bounce)
            elif (fish_smoothed[i] < 1.5 and fish_smoothed[i-1] >= 1.5 and
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Fisher crosses above +1.5 (overbought) or trend fails
            if (fish_smoothed[i] >= 1.5 or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Fisher crosses below -1.5 (oversold) or trend fails
            if (fish_smoothed[i] <= -1.5 or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals