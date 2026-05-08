#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Price_Action_Structure_1dTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and price action structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for additional trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily swing points for price action structure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Identify daily swing highs and lows
    swing_high = np.zeros_like(high_1d)
    swing_low = np.zeros_like(low_1d)
    
    for i in range(2, len(high_1d) - 2):
        # Swing high: higher than 2 bars on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = high_1d[i]
        # Swing low: lower than 2 bars on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = low_1d[i]
    
    # Forward fill swing points to get structure levels
    for i in range(1, len(swing_high)):
        if swing_high[i] == 0:
            swing_high[i] = swing_high[i-1]
        if swing_low[i] == 0:
            swing_low[i] = swing_low[i-1]
    
    # Structure levels: recent swing high for resistance, swing low for support
    structure_resistance = swing_high
    structure_support = swing_low
    
    # Align structure levels to 12h timeframe
    structure_resistance_aligned = align_htf_to_ltf(prices, df_1d, structure_resistance)
    structure_support_aligned = align_htf_to_ltf(prices, df_1d, structure_support)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA34 for higher timeframe trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation - 12-period average volume (6h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(structure_resistance_aligned[i]) or np.isnan(structure_support_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above structure resistance + above 1d EMA34 + above 1w EMA34 + volume confirmation
            if (close[i] > structure_resistance_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                close[i] > ema_34_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below structure support + below 1d EMA34 + below 1w EMA34 + volume confirmation
            elif (close[i] < structure_support_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  close[i] < ema_34_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below structure support OR below 1d EMA34
            if close[i] < structure_support_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above structure resistance OR above 1d EMA34
            if close[i] > structure_resistance_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals