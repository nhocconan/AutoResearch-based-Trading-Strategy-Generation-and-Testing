#!/usr/bin/env python3
"""
6h_cci_momentum_1d_trend_volume_v1
Hypothesis: CCI(20) captures momentum on 6h. Long when CCI > 100 and price above 1d EMA200 (uptrend). Short when CCI < -100 and price below 1d EMA200 (downtrend). Volume confirmation filters weak signals. Works in bull/bear by following higher timeframe trend. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_momentum_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d EMA200 for trend filter
    ema_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    
    # Align 1d EMA200 to 6h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # CCI(20) on 6h: (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(cci[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: CCI drops below 100 or price breaks below EMA200
            if cci[i] < 100 or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: CCI rises above -100 or price breaks above EMA200
            if cci[i] > -100 or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI > 100, with volume and price above EMA200
            if (cci[i] > 100 and vol_confirm and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI < -100, with volume and price below EMA200
            elif (cci[i] < -100 and vol_confirm and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals