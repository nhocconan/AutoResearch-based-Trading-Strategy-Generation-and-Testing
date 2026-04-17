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
    
    # Get weekly data for trend (to avoid look-ahead, we use previous week's close)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly trend: price above/below previous week's close
    weekly_trend = np.where(close_1w > np.roll(close_1w, 1), 1, -1)
    weekly_trend[0] = 0  # first value has no previous
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Daily volatility filter: ATR(14) > 20-period average
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need ATR MA20 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(volume_ma20[i]) or
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period average (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: weekly uptrend + volatility + volume
            if weekly_trend_aligned[i] == 1 and volatility_filter and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + volatility + volume
            elif weekly_trend_aligned[i] == -1 and volatility_filter and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or volatility drops
            if weekly_trend_aligned[i] == -1 or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or volatility drops
            if weekly_trend_aligned[i] == 1 or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_Volume_Volatility"
timeframe = "1d"
leverage = 1.0