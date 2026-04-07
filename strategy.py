#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h Trend and Daily Volume Filter
# Hypothesis: In trending markets, pullbacks to the 21 EMA on 1h offer high-probability entries when aligned with 4h EMA trend and elevated daily volume. Works in bull via long pullbacks and in bear via short pullbacks. Volume filter ensures institutional participation. Target: 15-37 trades/year.
name = "1h_ema_pullback_4h_trend_daily_volume_v1"
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
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Calculate 1h EMA(21)
    ema_1h = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema_1h[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Volume filter: current volume > daily average
        vol_filter = volume[i] > vol_ma_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1h EMA
            if close[i] < ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above 1h EMA
            if close[i] > ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price near 1h EMA in uptrend + volume
            if uptrend and vol_filter and close[i] <= ema_1h[i] * 1.005 and close[i] >= ema_1h[i] * 0.995:
                position = 1
                signals[i] = 0.20
            # Enter short: price near 1h EMA in downtrend + volume
            elif downtrend and vol_filter and close[i] >= ema_1h[i] * 0.995 and close[i] <= ema_1h[i] * 1.005:
                position = -1
                signals[i] = -0.20
    
    return signals