#!/usr/bin/env python3
"""
1h_ema_cci_pullback_4d_trend_volume_v1
Hypothesis: On 1h timeframe, use CCI(14) pullbacks to 4h EMA(50) for entries, filtered by 1d EMA(200) trend and volume confirmation.
In bull markets, pullbacks to rising 4h EMA during uptrend (1d EMA200 up) offer long entries.
In bear markets, pullbacks to declining 4h EMA during downtrend (1d EMA200 down) offer short entries.
Volume confirms genuine pullback/respect of dynamic support/resistance.
Target: 15-37 trades/year (~60-150 total over 4 years) by using strict pullback conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_cci_pullback_4d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    ema_200_1d = df_1d['close'].ewm(span=200, adjust=False).mean()
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d.values)
    
    # 4h data for EMA50 and CCI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for dynamic support/resistance
    ema_50_4h = df_4h['close'].ewm(span=50, adjust=False).mean()
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h.values)
    
    # CCI(14) on 1h: (Typical Price - SMA(TP,14)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=14, min_periods=14).mean()
    mad = tp_series.rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation (20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: 1d EMA200 slope (use 5-period change)
        ema_200_slope = ema_200_1d_aligned[i] - ema_200_1d_aligned[i-5] if i >= 5 else 0
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h EMA50 or CCI drops below -50
            if close[i] < ema_50_4h_aligned[i] or cci[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price breaks above 4h EMA50 or CCI rises above 50
            if close[i] > ema_50_4h_aligned[i] or cci[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: CCI between -50 and 50 (pullback zone), price near 4h EMA50, uptrend, volume
            if (cci[i] > -50 and cci[i] < 50 and
                abs(close[i] - ema_50_4h_aligned[i]) < (ema_50_4h_aligned[i] * 0.005) and  # within 0.5% of EMA
                ema_200_slope > 0 and  # 1d uptrend
                vol_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: CCI between -50 and 50 (pullback zone), price near 4h EMA50, downtrend, volume
            elif (cci[i] > -50 and cci[i] < 50 and
                  abs(close[i] - ema_50_4h_aligned[i]) < (ema_50_4h_aligned[i] * 0.005) and  # within 0.5% of EMA
                  ema_200_slope < 0 and  # 1d downtrend
                  vol_confirm):
                position = -1
                signals[i] = -0.20
    
    return signals