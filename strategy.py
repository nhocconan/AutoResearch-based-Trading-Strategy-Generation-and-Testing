# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Strategy: 12h timeframe using Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation
# Hypothesis: Price breaking above/below Camarilla R1/S1 levels on 12h chart indicates breakout when aligned with 1d trend and volume spike. 
# Works in bull (breakouts with trend) and bear (mean reversion off extremes) due to trend filter.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once for Camarilla pivots, EMA trend, and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla R1 and S1 (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 for previous day
    p = (high_1d + low_1d + close_1d_vals) / 3
    r1 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    s1 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    
    # Daily EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d_vals)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily ATR(14) for volatility filter (ensure volatility present)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d_vals[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d_vals[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Camarilla levels, EMA, and ATR to 12h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (price > EMA34) + volatility + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                atr_1d_aligned[i] > 0 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (price < EMA34) + volatility + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  atr_1d_aligned[i] > 0 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 (mean reversion)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 (mean reversion)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals