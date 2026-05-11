#!/usr/bin/env python3
name = "1d_WeeklyBreakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly high/low for breakout levels
    weekly_high = high_weekly
    weekly_low = low_weekly
    
    # Align weekly levels to daily
    weekly_high_daily = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_daily = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Weekly EMA20 for trend filter
    close_weekly_series = pd.Series(close_weekly)
    ema_weekly = close_weekly_series.ewm(span=20, min_periods=20).mean().values
    ema_weekly_daily = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Volatility filter: ATR > 0.6 * ATR(50)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma * 0.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_daily[i]) or np.isnan(weekly_low_daily[i]) or 
            np.isnan(ema_weekly_daily[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high AND above weekly EMA20 (uptrend) AND volume spike AND volatility present
            if close[i] > weekly_high_daily[i] and close[i] > ema_weekly_daily[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low AND below weekly EMA20 (downtrend) AND volume spike AND volatility present
            elif close[i] < weekly_low_daily[i] and close[i] < ema_weekly_daily[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly low OR below weekly EMA20 (trend change)
            if close[i] < weekly_low_daily[i] or close[i] < ema_weekly_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly high OR above weekly EMA20 (trend change)
            if close[i] > weekly_high_daily[i] or close[i] > ema_weekly_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals