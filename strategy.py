#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d/1w Donchian channel breakout + volume confirmation + ATR volatility filter
# Strategy: Long when price breaks above 20-period high with volume > 1.5x average and ATR > threshold
# Short when price breaks below 20-period low with volume > 1.5x average and ATR > threshold
# Uses multi-timeframe Donchian channels (1d for higher timeframe context, 12h for entry)
# Volume surge confirms breakout strength, ATR filter avoids low volatility false breakouts
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag in ranging markets
# Works in both bull/bear markets via volatility-adjusted breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe Donchian context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for ultra-higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h Donchian channel (20-period) for entry signals
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA (50-period) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Donchian levels to 12h timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_1d_aligned[i]) or 
            np.isnan(low_20_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility environments
        volatility_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Higher timeframe trend filter: only trade in direction of 1w trend
        uptrend_filter = close[i] > ema_50_1w_aligned[i]
        downtrend_filter = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions with confirmation
        long_breakout = close[i] > high_20[i] and close[i-1] <= high_20[i-1]
        short_breakout = close[i] < low_20[i] and close[i-1] >= low_20[i-1]
        
        # Entry logic with all filters
        long_entry = (long_breakout and 
                     volume_surge and 
                     volatility_filter and 
                     uptrend_filter)
        
        short_entry = (short_breakout and 
                      volume_surge and 
                      volatility_filter and 
                      downtrend_filter)
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (close[i] < low_20[i] or atr[i] < 0.005 * close[i])
        exit_short = position == -1 and (close[i] > high_20[i] or atr[i] < 0.005 * close[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_donchian_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0