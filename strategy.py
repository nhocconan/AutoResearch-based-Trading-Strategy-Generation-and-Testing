#!/usr/bin/env python3
# 4h_1d_cci_momentum_v1
# Strategy: 4-hour CCI momentum with volume confirmation and 1-day trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI captures cyclical momentum with clear overbought/oversold levels.
# Bullish when CCI crosses above -100 from below with volume confirmation (VOL > 1.5x 20-period average) and price above 1-day EMA50.
# Bearish when CCI crosses below +100 from above with volume confirmation and price below 1-day EMA50.
# Works in bull markets by catching pullbacks in uptrend and in bear markets by catching bounces in downtrend.
# Uses tight entry conditions to limit trades (~25-40/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h CCI (20-period)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    ma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    cci = (typical_price - ma_tp) / (0.015 * mad)
    cci = cci.fillna(0).values
    
    # 4h Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # CCI crosses: long when crossing above -100, short when crossing below +100
        cci_cross_up = cci[i-1] <= -100 and cci[i] > -100
        cci_cross_down = cci[i-1] >= 100 and cci[i] < 100
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: CCI cross + volume + trend alignment
        if cci_cross_up and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif cci_cross_down and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite CCI cross with volume confirmation
        elif position == 1 and cci_cross_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_cross_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals