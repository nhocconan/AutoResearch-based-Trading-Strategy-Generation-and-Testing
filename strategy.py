#!/usr/bin/env python3
# 1d_1w_cci_volume_breakout_v2
# Strategy: Daily CCI momentum with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: CCI captures cyclical momentum with weekly trend filter (weekly EMA50) for trend alignment.
# Long when CCI crosses above +100 with volume > 1.5x 20-day average and price above weekly EMA50.
# Short when CCI crosses below -100 with volume > 1.5x 20-day average and price below weekly EMA50.
# Exit on opposite CCI cross with volume confirmation. Uses tight conditions to limit trades (~15-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cci_volume_breakout_v2"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily CCI (20-period)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - sma_tp) / (0.015 * mad)
    cci = cci.fillna(0).values
    
    # Daily Volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # CCI crosses
        cci_cross_up = cci[i-1] <= 100 and cci[i] > 100
        cci_cross_down = cci[i-1] >= -100 and cci[i] < -100
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
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