#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume_Regime"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Time filter: 08-20 UTC (reduces noise)
    hours = pd.DatetimeIndex(open_time).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla pivot levels (from previous 4h bar)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4.0
    s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Volatility regime filter: ATR-based to avoid choppy and excessively volatile markets
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr2 = np.maximum(np.absolute(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[tr1[0]], tr2]) if len(tr1) > 0 else np.array([0.0])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_pct = atr / close
    # Only trade when volatility is moderate (not too low, not too high)
    vol_regime = (atr_pct > 0.010) & (atr_pct < 0.040)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside trading hours
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i]) or
            not time_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R3 + above 4h EMA34 + volume filter + vol regime
            if high[i] > r3_4h_aligned[i] and close[i] > ema_34_4h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakdown below S3 + below 4h EMA34 + volume filter + vol regime
            elif low[i] < s3_4h_aligned[i] and close[i] < ema_34_4h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: breakdown below S3 or below 4h EMA34
            if low[i] < s3_4h_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: breakout above R3 or above 4h EMA34
            if high[i] > r3_4h_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals