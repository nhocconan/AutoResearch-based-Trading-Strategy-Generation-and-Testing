#!/usr/bin/env python3
# 4h_KAMA_Trend_Reverse_With_Volume
# Hypothesis: In trending markets (12h EMA50), enter long on KAMA upturn after pullback and volume confirmation,
# and short on KAMA downturn after bounce and volume confirmation. Uses KAMA for adaptive trend following,
# volume spike for confirmation, and avoids choppy markets. Designed for both bull and bear markets by
# following the 12h trend. Targets ~25 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_Reverse_With_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, n=1, prepend=close[0])), axis=0)
    # Fix volatility calculation using rolling sum
    volatility_rolling = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 for upturn, -1 for downturn
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0  # first value undefined
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for KAMA and indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(kama_dir[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA upturn in uptrend with volume spike
            if (kama_dir[i] == 1 and
                trend_12h_up_aligned[i] > 0.5 and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA downturn in downtrend with volume spike
            elif (kama_dir[i] == -1 and
                  trend_12h_down_aligned[i] > 0.5 and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA downturn or trend change
            if (kama_dir[i] == -1 or
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA upturn or trend change
            if (kama_dir[i] == 1 or
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals