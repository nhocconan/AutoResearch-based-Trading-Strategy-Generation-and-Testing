#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_VolumeSpike_4hTrend_1dFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA20 for trend direction
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for higher timeframe filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (1h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # RSI for momentum confirmation (14-period)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Strong volume spike
        rsi_ok_long = rsi_values[i] > 50 and rsi_values[i] < 70  # Bullish momentum
        rsi_ok_short = rsi_values[i] < 50 and rsi_values[i] > 30  # Bearish momentum
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price above 4h EMA (uptrend) + above 1d EMA (strong trend) + volume spike + RSI bullish
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_ok and 
                rsi_ok_long and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA (downtrend) + below 1d EMA (strong trend) + volume spike + RSI bearish
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_ok and 
                  rsi_ok_short and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h EMA or volume dries up
            if close[i] < ema_4h_aligned[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h EMA or volume dries up
            if close[i] > ema_4h_aligned[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals