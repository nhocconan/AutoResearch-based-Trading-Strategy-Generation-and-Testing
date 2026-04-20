#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_20_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h: Trend filter (EMA50) ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 4h: Price and volume ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        ema_val = ema50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian + uptrend + volume confirmation
            if (high_val > high_max_20[i] and      # Breakout above 20-period high
                close_val > ema_val and            # Price above 12h EMA50 (uptrend)
                vol_ratio_val > 1.5):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian + downtrend + volume confirmation
            elif (low_val < low_min_20[i] and      # Breakdown below 20-period low
                  close_val < ema_val and          # Price below 12h EMA50 (downtrend)
                  vol_ratio_val > 1.5):            # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below lower Donchian or trend reversal
            if (low_val < low_min_20[i] or         # Break below 20-period low
                close_val < ema_val):              # Price below 12h EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above upper Donchian or trend reversal
            if (high_val > high_max_20[i] or       # Break above 20-period high
                close_val > ema_val):              # Price above 12h EMA50 (trend reversal)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals