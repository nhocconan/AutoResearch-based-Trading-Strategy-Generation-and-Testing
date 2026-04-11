#!/usr/bin/env python3
# 12h_1d_keltner_breakout_volume_v1
# Strategy: 12h Keltner Channel breakout with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansion with trend direction.
# In bull markets, break above upper channel with volume; in bear markets, break below lower channel.
# Uses 1-day EMA50 for trend filter to avoid counter-trend trades. Low frequency (~15-30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Keltner Channel (20, 2) on 12h timeframe
    # Typical Price = (High + Low + Close) / 3
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    
    # EMA of Typical Price (20-period)
    ema_tp = tp_series.ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Average True Range (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_series = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Keltner Bands
    keltner_upper = ema_tp + (2.0 * atr_series)
    keltner_lower = ema_tp - (2.0 * atr_series)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(keltner_upper.iloc[i]) or np.isnan(keltner_lower.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Keltner breakout + volume + trend alignment
        if (close[i] > keltner_upper.iloc[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < keltner_lower.iloc[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle (EMA of typical price) or trend change
        elif position == 1 and (close[i] <= ema_tp.iloc[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= ema_tp.iloc[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals