#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
    # Long: price breaks above 20-day high + volume > 1.5x 20-day avg + 1w close > 1w EMA21
    # Short: price breaks below 20-day low + volume > 1.5x 20-day avg + 1w close < 1w EMA21
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 1d optimal range (80-200 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period Donchian channels on 1d data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 1d timeframe
    atr_1d = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_1d[i] = tr  # Simple average for warmup
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: 1d close above/below EMA21 (from 1w)
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > highest_20[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < lowest_20[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_1d[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0