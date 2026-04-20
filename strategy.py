#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE
    df_week = get_htf_data(prices, '1w')
    week_high = df_week['high'].values
    week_low = df_week['low'].values
    week_close = df_week['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = week_high[1:] - week_low[1:]
    tr2 = np.abs(week_high[1:] - week_close[:-1])
    tr3 = np.abs(week_low[1:] - week_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_week = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA(50) and EMA(200)
    week_close_series = pd.Series(week_close)
    ema_50_week = week_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_week = week_close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to 12h (wait for weekly bar close)
    ema_50_week_aligned = align_htf_to_ltf(prices, df_week, ema_50_week)
    ema_200_week_aligned = align_htf_to_ltf(prices, df_week, ema_200_week)
    atr_14_week_aligned = align_htf_to_ltf(prices, df_week, atr_14_week)
    
    # Calculate 12h price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in weekly aligned data
        if np.isnan(ema_50_week_aligned[i]) or np.isnan(ema_200_week_aligned[i]) or np.isnan(atr_14_week_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_week_aligned[i]
        ema_200_val = ema_200_week_aligned[i]
        atr_val = atr_14_week_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price above weekly EMA200 and low volatility (below 30th percentile)
            if price > ema_200_val and atr_val < np.nanpercentile(atr_14_week_aligned[:i+1], 30):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA50 and low volatility (below 30th percentile)
            elif price < ema_50_val and atr_val < np.nanpercentile(atr_14_week_aligned[:i+1], 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly EMA50 or volatility spikes above 50th percentile
            if price < ema_50_val or atr_val > np.nanpercentile(atr_14_week_aligned[:i+1], 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly EMA200 or volatility spikes above 50th percentile
            if price > ema_200_val or atr_val > np.nanpercentile(atr_14_week_aligned[:i+1], 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA50_EMA200_WeeklyVolatilityFilter"
timeframe = "12h"
leverage = 1.0