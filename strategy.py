#!/usr/bin/env python3
# 6h_1d_cci_volume_reversal_v1
# Strategy: 6-hour CCI with volume spike and 1-day trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. In ranging markets, price reverses from extremes with volume confirmation. Trend filter avoids counter-trend trades. Works in both bull/bear by adapting to regime via CCI extremes and volume spikes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_volume_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h CCI(20)
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - tp_mean) / (0.015 * tp_dev)
    cci = cci.values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: CCI reversal + volume + trend alignment
        if cci[i] < -100 and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif cci[i] > 100 and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone
        elif position == 1 and cci[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] < 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals