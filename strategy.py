#!/usr/bin/env python3
# 6h_Stochastic_Pullback_1dTrend_VolumeFilter
# Hypothesis: Stochastic oscillator identifies oversold/overbought conditions on 6h timeframe.
# Entries occur when price pulls back to 1d EMA20 during extreme Stoch readings, confirmed by volume spike.
# Works in bull markets via buying oversold dips in uptrends, and bear markets via selling overbought rallies in downtrends.
# Uses 1d trend filter and volume confirmation to avoid chop, targeting 15-35 trades/year.

name = "6h_Stochastic_Pullback_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d trend filter (EMA20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1d_up = close_1d > ema20_1d
    trend_1d_down = close_1d < ema20_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Stochastic oscillator (14,3,3) on 6h
    lookback = 14
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            continue
        start_idx = i - lookback + 1
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    stoch_k = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] == lowest_low[i] or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            stoch_k[i] = 50.0
        else:
            stoch_k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
    
    stoch_smooth = np.full(n, np.nan)
    smooth_sum = 0
    for i in range(n):
        smooth_sum += stoch_k[i]
        if i >= 3:
            smooth_sum -= stoch_k[i-3]
        if i >= 2:
            stoch_smooth[i] = smooth_sum / 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(stoch_smooth[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Stoch oversold (<20), price near 1d EMA20, volume confirmation, 1d uptrend
            if (stoch_smooth[i] < 20 and
                low[i] <= ema20_1d[i] * 1.005 and  # within 0.5% above EMA
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Stoch overbought (>80), price near 1d EMA20, volume confirmation, 1d downtrend
            elif (stoch_smooth[i] > 80 and
                  high[i] >= ema20_1d[i] * 0.995 and  # within 0.5% below EMA
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Stoch overbought (>80) or 1d trend turns down
            if (stoch_smooth[i] > 80 or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Stoch oversold (<20) or 1d trend turns up
            if (stoch_smooth[i] < 20 or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals