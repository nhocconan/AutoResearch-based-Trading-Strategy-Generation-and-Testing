#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(20) for trend
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(14) for volatility
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # 12h time filters: 00:00-08:00 and 16:00-24:00 UTC (avoid 08:00-16:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Time filter: avoid 08:00-16:00 UTC (low volatility period)
        hour = hours[i]
        in_filter = (hour < 8) or (hour >= 16)
        
        if not in_filter:
            # Outside preferred hours: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema_20_aligned[i]
        trend_down = close[i] < ema_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_14_aligned[i] > 0.5 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        
        # Entry conditions - selective to reduce trades
        long_entry = trend_up and vol_filter
        short_entry = trend_down and vol_filter
        
        # Exit conditions: opposite trend or volatility spike
        atr_ma = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        volatility_spike = atr_14_aligned[i] > 3.0 * atr_ma[i]
        long_exit = not trend_up or volatility_spike
        short_exit = not trend_down or volatility_spike
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyEMA20_TimeFilter"
timeframe = "12h"
leverage = 1.0