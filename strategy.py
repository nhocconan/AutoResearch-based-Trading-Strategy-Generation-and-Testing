#!/usr/bin/env python3
"""
12h_1d_cci_breakout_volume_v1
Hypothesis: 12h CCI(20) with 1d trend filter (EMA50) and volume confirmation.
Long when CCI crosses above +100 in uptrend, short when CCI crosses below -100 in downtrend.
Works in bull (buy breakouts above +100) and bear (sell breakdowns below -100).
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h CCI(20)
    tp = (high + low + close) / 3
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean()
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - ma_tp.values) / (0.015 * md.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 or trend changes
            if cci[i] < 100 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 or trend changes
            if cci[i] > -100 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: CCI crosses above +100 with volume and uptrend
            if (cci[i] > 100 and cci[i-1] <= 100 and  # Cross above
                close[i] > ema_50_1d_aligned[i] and   # Uptrend
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below -100 with volume and downtrend
            elif (cci[i] < -100 and cci[i-1] >= -100 and  # Cross below
                  close[i] < ema_50_1d_aligned[i] and   # Downtrend
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals