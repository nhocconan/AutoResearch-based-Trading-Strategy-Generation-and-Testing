#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1d trend filter
    # Long: price breaks above Donchian upper band + volume > 1.5x 20-period 1d average + 1d close > 1d EMA100
    # Short: price breaks below Donchian lower band + volume > 1.5x 20-period 1d average + 1d close < 1d EMA100
    # Uses discrete sizing (0.30) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range (80-200 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 4h data (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate volume average for confirmation (using 1d data)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_upper_aligned = donchian_upper  # Already on 4h
    donchian_lower_aligned = donchian_lower  # Already on 4h
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.30  # 30% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 4h timeframe
    atr_4h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_4h[i] = tr  # Simple average for warmup
        else:
            atr_4h[i] = 0.93 * atr_4h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_100_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Find the corresponding 1d bar index for current 4h bar
        volume_confirmed = False
        if i >= 16:  # At least one full 1d bar has passed (16*4h = 4d)
            # Get the most recent completed 1d bar
            daily_index = i // 16
            if daily_index < len(volume_1d) and daily_index < len(vol_avg_20_1d_aligned):
                volume_confirmed = volume_1d[daily_index] > 1.5 * vol_avg_20_1d_aligned[daily_index]
        
        # Trend filter: 4h close above/below EMA100 (from 1d)
        uptrend = close[i] > ema_100_1d_aligned[i]
        downtrend = close[i] < ema_100_1d_aligned[i]
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_upper_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < donchian_lower_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.5 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.5 * atr_4h[i]
        
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

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0