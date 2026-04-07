#!/usr/bin/env python3
"""
12h_cci_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use CCI(20) breakouts for entry signals, filtered by 1d EMA trend and volume confirmation. 
In bull markets, CCI > 100 captures momentum; in bear markets, CCI < -100 captures short opportunities. 
Volume confirms genuine breakouts. 1d EMA filter ensures alignment with higher timeframe trend, reducing whipsaw.
Target: 12-37 trades/year (~50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_cci_breakout_1d_trend_volume_v1"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # CCI(20) on 12h: (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: CCI drops below 0 or price breaks below EMA50
            if cci[i] < 0 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: CCI rises above 0 or price breaks above EMA50
            if cci[i] > 0 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI > 100, with volume and price above EMA50
            if (cci[i] > 100 and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI < -100, with volume and price below EMA50
            elif (cci[i] < -100 and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals