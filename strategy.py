#!/usr/bin/env python3
# 4h_Bollinger_Bandwidth_Reversal_with_1dTrend_Volume
# Hypothesis: Bollinger Bandwidth contraction indicates low volatility; expansion signals breakout.
# Use 1d trend filter to avoid counter-trend trades, and volume spike for confirmation.
# Works in bull (breakouts with trend) and bear (mean reversion at band extremes with volume).
# Tight entries to avoid overtrading.

name = "4h_Bollinger_Bandwidth_Reversal_with_1dTrend_Volume"
timeframe = "4h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Bollinger Bands (20, 2) on 4h
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean()
    std20 = close_series.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bandwidth = (upper - lower) / sma20  # Bandwidth as percentage
    
    # Bandwidth percentile (50-period) for regime detection
    bandwidth_series = bandwidth.values
    bandwidth_percentile = pd.Series(bandwidth_series).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(bandwidth_percentile[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        # Low bandwidth = low volatility (squeeze), high bandwidth = expansion
        bw = bandwidth_percentile[i]
        
        if position == 0:
            # Long: bandwidth expansion (breakout) with uptrend and volume spike
            if (bw > 0.8 and  # High bandwidth percentile (expansion)
                trend_1d_up_aligned[i] > 0.5 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bandwidth expansion (breakdown) with downtrend and volume spike
            elif (bw > 0.8 and 
                  trend_1d_down_aligned[i] > 0.5 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            # Mean reversion in range: price at lower band with volume spike
            elif (bw < 0.3 and  # Low bandwidth (squeeze/range)
                  close[i] <= lower[i] and 
                  volume_spike):
                signals[i] = 0.25
                position = 1
            # Mean reversion in range: price at upper band with volume spike
            elif (bw < 0.3 and 
                  close[i] >= upper[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bandwidth contraction (mean reversion) or trend fails
            if (bw < 0.3 or  # Contraction back to range
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bandwidth contraction (mean reversion) or trend fails
            if (bw < 0.3 or  # Contraction back to range
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals