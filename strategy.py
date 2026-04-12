#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 20:
        return np.zeros(n)
    
    close_week = df_week['close'].values
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    
    # Calculate weekly SMA(50) for trend
    close_week_series = pd.Series(close_week)
    sma_50_week = close_week_series.rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility
    df_day = get_htf_data(prices, '1d')
    if len(df_day) < 30:
        return np.zeros(n)
    
    high_day = df_day['high'].values
    low_day = df_day['low'].values
    close_day = df_day['close'].values
    
    tr1 = np.abs(high_day - low_day)
    tr2 = np.abs(high_day - np.roll(close_day, 1))
    tr3 = np.abs(low_day - np.roll(close_day, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_day = np.full(len(df_day), np.nan)
    for i in range(14, len(df_day)):
        atr_day[i] = np.mean(tr[i-14:i+1])
    
    # Align weekly SMA and daily ATR to daily timeframe
    sma_50_week_aligned = align_htf_to_ltf(prices, df_week, sma_50_week)
    atr_day_aligned = align_htf_to_ltf(prices, df_day, atr_day)
    
    # Calculate daily ATR moving average (20)
    atr_day_series = pd.Series(atr_day)
    atr_ma_20_day = atr_day_series.rolling(window=20, min_periods=20).mean().values
    atr_ma_20_day_aligned = align_htf_to_ltf(prices, df_day, atr_ma_20_day)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma_50_week_aligned[i]) or np.isnan(atr_day_aligned[i]) or 
            np.isnan(atr_ma_20_day_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA(50)
        trend_up = close[i] > sma_50_week_aligned[i]
        trend_down = close[i] < sma_50_week_aligned[i]
        
        # Volatility filter: daily ATR > 0.5 * its 20-period MA (avoid low volatility)
        vol_filter = atr_day_aligned[i] > 0.5 * atr_ma_20_day_aligned[i]
        
        # Entry conditions
        long_entry = trend_up and vol_filter
        short_entry = trend_down and vol_filter
        
        # Exit conditions: trend reversal
        long_exit = trend_down
        short_exit = trend_up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_sma50_trend_vol_filter"
timeframe = "1d"
leverage = 1.0