#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w trend filter and 1d volume confirmation
    # Long: price breaks above 20-period high + 1w close > 1w EMA200 + 1d volume > 1.5x 20-period average
    # Short: price breaks below 20-period low + 1w close < 1w EMA200 + 1d volume > 1.5x 20-period average
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR using true range approximation for 6h timeframe
    atr_6h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_6h[i] = tr  # Simple average for warmup
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period 1d volume average
        vol_avg_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_6h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_6h[i]
        
        # Trend filter: 6h close above/below 1w EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_high[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < donchian_low[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
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

name = "6h_1w_1d_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0