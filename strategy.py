#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_cci_trend_follow
# Uses daily CCI to determine trend direction and 6h CCI for entry timing.
# Long when daily CCI > 100 (bullish trend) and 6h CCI crosses above -100 (pullback end).
# Short when daily CCI < -100 (bearish trend) and 6h CCI crosses below +100 (pullback end).
# Uses volume confirmation: volume > 1.5 * 20-period average to avoid low-volume breakouts.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drift.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell bounces in downtrend).

name = "6h_1d_cci_trend_follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI (20-period)
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp_1d = tp_1d.rolling(window=20, min_periods=20).mean()
    mad_1d = tp_1d.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_1d = (tp_1d - sma_tp_1d) / (0.015 * mad_1d)
    cci_1d_values = cci_1d.values
    
    # Align daily CCI to 6h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d_values)
    
    # Calculate 6h CCI (20-period) for entry timing
    tp = (high + low + close) / 3
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if values not ready
        if np.isnan(cci_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Long setup: daily CCI > 100 (uptrend) and 6h CCI crosses above -100 (end of pullback)
        if cci_1d_aligned[i] > 100 and cci[i] > -100 and cci[i-1] <= -100:
            if vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        # Short setup: daily CCI < -100 (downtrend) and 6h CCI crosses below +100 (end of bounce)
        elif cci_1d_aligned[i] < -100 and cci[i] < 100 and cci[i-1] >= 100:
            if vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        # Exit: opposite trend condition
        elif (cci_1d_aligned[i] < -100 and position == 1) or (cci_1d_aligned[i] > 100 and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals