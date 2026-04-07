#!/usr/bin/env python3
"""
4h_cci_breakout_12h_trend_volume_v2
Hypothesis: On 4h timeframe, use CCI(20) breakouts for entry signals, filtered by 12h EMA trend and volume confirmation. 
In bull markets, CCI > 100 captures momentum; in bear markets, CCI < -100 captures short opportunities. 
Volume confirms genuine breakouts. 12h EMA filter ensures alignment with higher timeframe trend, reducing whipsaw.
Tightened entry conditions (reduced from 3 to 2) to lower trade frequency and avoid overtrading.
Target: 15-30 trades/year (~60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_12h_trend_volume_v2"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter (more responsive than 200 for 4h trading)
    ema_50 = df_12h['close'].ewm(span=50, adjust=False).mean()
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50.values)
    
    # CCI(20) on 4h: (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation (20-period average on 4h)
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