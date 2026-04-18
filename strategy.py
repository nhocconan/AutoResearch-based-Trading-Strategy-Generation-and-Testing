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
    
    # Get weekly data for long-term trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for price levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA34 for trend direction
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily high/low for support/resistance levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    # Use 20-period high/low from daily for entry levels
    high_20d = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # 6-period ATR for volatility filter (matches 6h timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr6 = pd.Series(tr).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # wait for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or
            np.isnan(high_20d_aligned[i]) or
            np.isnan(low_20d_aligned[i]) or
            np.isnan(atr6[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma = pd.Series(atr6).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr6[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        if position == 0:
            # Long: price breaks above 20-day high with weekly uptrend, volume and volatility
            if (close[i] > high_20d_aligned[i-1] and 
                close[i] > ema34_1w_aligned[i] and  # price above weekly EMA34 (uptrend)
                volume_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with weekly downtrend, volume and volatility
            elif (close[i] < low_20d_aligned[i-1] and 
                  close[i] < ema34_1w_aligned[i] and  # price below weekly EMA34 (downtrend)
                  volume_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or loses weekly uptrend
            if close[i] < low_20d_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or loses weekly downtrend
            if close[i] > high_20d_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6d_WeeklyEMA34_Daily20_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0