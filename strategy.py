#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(15) breakout with 1d volume confirmation and 1d trend filter
    # Long: price breaks above 15-period high + volume > 1.3x 15-period avg + 1d close > 1d EMA20
    # Short: price breaks below 15-period low + volume > 1.3x 15-period avg + 1d close < 1d EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-30 trades/year to stay within 12h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation, Donchian calculation, and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (15-period)
    high_15 = pd.Series(high_1d).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(low_1d).rolling(window=15, min_periods=15).min().values
    
    # Calculate 1d volume average (15-period) for confirmation
    vol_avg_15_1d = pd.Series(volume_1d).rolling(window=15, min_periods=15).mean().values
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    high_15_aligned = align_htf_to_ltf(prices, df_1d, high_15)
    low_15_aligned = align_htf_to_ltf(prices, df_1d, low_15)
    vol_avg_15_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_15_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_1d = np.zeros(n)  # Simplified ATR using daily range
    
    # Calculate simplified ATR (daily range) for stoploss
    for i in range(n):
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d < len(high_1d) and idx_1d < len(low_1d):
            daily_range = high_1d[idx_1d] - low_1d[idx_1d]
            atr_1d[i] = daily_range * 0.5  # Approximate ATR as 50% of daily range
        else:
            atr_1d[i] = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_15_aligned[i]) or 
            np.isnan(low_15_aligned[i]) or
            np.isnan(vol_avg_15_1d_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 15-period average
        idx_1d = i // 2  # 12h bars in 1d timeframe (2 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_15_1d_aligned[i]
        
        # Trend filter: 1d close above/below EMA20
        uptrend = close_1d[idx_1d] > ema_20_1d_aligned[i] if idx_1d < len(close_1d) else False
        downtrend = close_1d[idx_1d] < ema_20_1d_aligned[i] if idx_1d < len(close_1d) else False
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > high_15_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < low_15_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 1.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 1.5 * atr_1d[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 1.5 * atr_1d[i]
        
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

name = "12h_1d_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0