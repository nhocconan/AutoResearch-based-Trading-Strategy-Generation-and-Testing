#!/usr/bin/env python3
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
    
    # Get 4h data for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA 20 for trend direction
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h ATR for volatility filter
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1h Donchian channels (10-period for faster entry)
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume filter: volume > 1.5x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 8-20 UTC (intraday active hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA20
        price_above_ema = close[i] > ema_20_4h_aligned[i]
        price_below_ema = close[i] < ema_20_4h_aligned[i]
        
        # Volatility filter: current 1h ATR > 1.2x 4h ATR (avoid low volatility periods)
        atr_1h = np.abs(high[i] - low[i])
        vol_filter = atr_1h > (atr_4h_aligned[i] * 1.2)
        
        # Long conditions: price breaks above upper Donchian + above 4h EMA + volume + volatility
        long_breakout = (close[i] > highest_high[i-1] and price_above_ema and volume_filter[i] and vol_filter)
        # Short conditions: price breaks below lower Donchian + below 4h EMA + volume + volatility
        short_breakout = (close[i] < lowest_low[i-1] and price_below_ema and volume_filter[i] and vol_filter)
        
        if long_breakout:
            signals[i] = 0.20
            position = 1
        elif short_breakout:
            signals[i] = -0.20
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < lowest_low[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Donchian10_Breakout_4hEMA20_VolVolFilter_Session"
timeframe = "1h"
leverage = 1.0