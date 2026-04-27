#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.abs(high_1d[1:] - close_1d[:-1]), 
                       np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate daily range for breakout detection
    daily_range = high_1d - low_1d
    daily_range_avg = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    daily_range_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_range_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(daily_range_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume filter: current volume above 1d average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Breakout filter: price movement exceeds average daily range
        price_change = abs(close[i] - close[i-1])
        breakout_filter = price_change > (0.3 * daily_range_avg_aligned[i])
        
        # Long conditions: price above EMA34 + volume + volatility + breakout
        long_condition = (price_above_ema and 
                         volume_filter and 
                         vol_filter and 
                         breakout_filter)
        
        # Short conditions: price below EMA34 + volume + volatility + breakout
        short_condition = (price_below_ema and 
                          volume_filter and 
                          vol_filter and 
                          breakout_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_VolumeBreakout_4hTrend"
timeframe = "4h"
leverage = 1.0