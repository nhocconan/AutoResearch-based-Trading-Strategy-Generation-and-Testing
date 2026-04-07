#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h Trend and 1d Volume Filter
# Hypothesis: In trending markets (4h EMA alignment), pullbacks to 1h EMA offer high-probability entries.
# 1d volume filter ensures institutional participation. Works in bull (long pullbacks) and bear (short pullbacks).
# Target: 15-35 trades/year to avoid fee drag.
name = "1h_ema_pullback_4h_trend_1d_volume_v1"
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
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume SMA(20) for institutional participation filter
    vol_1d = df_1d['volume'].values
    vol_sma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # 1h EMA(21) for pullback entries
    close_s = pd.Series(close)
    ema_1h = close_s.ewm(span=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1h[i]) or 
            np.isnan(vol_sma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA slope (rising/falling)
        if i >= 51:
            ema_4h_prev = ema_4h_aligned[i-1]
            ema_4h_curr = ema_4h_aligned[i]
            trend_up = ema_4h_curr > ema_4h_prev
            trend_down = ema_4h_curr < ema_4h_prev
        else:
            trend_up = trend_down = False
        
        # Volume filter: current 1h volume > 1d average volume
        vol_filter = volume[i] > vol_sma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1h EMA (trend resumption failed)
            if close[i] < ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above 1h EMA (trend resumption failed)
            if close[i] > ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Enter long: 4h uptrend + price pulls back to/near 1h EMA + volume
            if trend_up and close[i] <= ema_1h[i] * 1.005 and vol_filter:
                position = 1
                signals[i] = 0.20
            # Enter short: 4h downtrend + price bounces to/near 1h EMA + volume
            elif trend_down and close[i] >= ema_1h[i] * 0.995 and vol_filter:
                position = -1
                signals[i] = -0.20
    
    return signals