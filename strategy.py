#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v1
# Hypothesis: Daily EMA(21) cross above/below weekly EMA(34) with volume confirmation.
# Weekly trend filter prevents counter-trend trades in volatile markets.
# Volume ensures participation, reducing false breakouts.
# Target: 15-25 trades/year with position size 0.25 to minimize fee drag.
# Works in bull markets (trend following) and bear markets (avoids counter-trend traps).

name = "1d_weekly_ema_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA(21)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly EMA(34) - get once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_ma[20:] = pd.Series(volume).rolling(window=20, min_periods=20).mean()[20:].values
    
    # Start from sufficient lookback
    start_idx = max(21, 20) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if EMA cross down or trend fails
            if ema_21[i] < ema_34_1w_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if EMA cross up or trend fails
            if ema_21[i] > ema_34_1w_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Daily EMA above weekly EMA with uptrend and volume
            if ema_21[i] > ema_34_1w_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Daily EMA below weekly EMA with downtrend and volume
            elif ema_21[i] < ema_34_1w_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals