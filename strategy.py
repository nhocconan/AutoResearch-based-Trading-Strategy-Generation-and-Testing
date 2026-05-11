# 1d_KAMA_Trend_Volume_1wTrend
# Hypothesis: Use 1d KAMA direction confirmed by 1w trend and volume spike for entries.
# Works in bull/bear via trend filter. Target: 15-30 trades/year to avoid fee drag.
# Uses price close for exits/reversals.

#!/usr/bin/env python3

name = "1d_KAMA_Trend_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- 1w Trend Filter: EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- 1d KAMA (ER=10) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # --- Volume Filter: spike above 1.5x median of last 20 days ---
    vol_median = pd.Series(volume_1d).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                # Hold position if no exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_1d[i] > ema34_1w_aligned[i]
        trend_down = close_1d[i] < ema34_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume_1d[i] > vol_threshold[i]
        
        if position == 0:
            # Enter long: price above KAMA + 1w uptrend + volume spike
            if close_1d[i] > kama[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close_1d[i]
            # Enter short: price below KAMA + 1w downtrend + volume spike
            elif close_1d[i] < kama[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close_1d[i]
        else:
            # Exit logic: reverse on opposite KAMA cross
            if position == 1:
                if close_1d[i] < kama[i]:
                    signals[i] = -0.25  # reverse to short
                    position = -1
                    entry_price = close_1d[i]
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_1d[i] > kama[i]:
                    signals[i] = 0.25   # reverse to long
                    position = 1
                    entry_price = close_1d[i]
                else:
                    signals[i] = -0.25
    
    return signals

# Note: No explicit stoploss; uses KAMA cross for exit to minimize whipsaw.
# Position size fixed at 0.25 to limit drawdown.