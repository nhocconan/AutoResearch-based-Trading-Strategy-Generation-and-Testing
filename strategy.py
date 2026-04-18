#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for long-term trend
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1_1d = df_1d['high'] - df_1d['low']
    tr2_1d = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3_1d = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h Donchian Channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR (14-period) for breakout filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h ATR Moving Average (20-period) for volatility filter
    atr_ma_6h = pd.Series(atr_6h).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(atr_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Trend filter: Price above/below 1d EMA200
        trend_up = close[i] > ema200_1d_aligned[i]
        trend_down = close[i] < ema200_1d_aligned[i]
        
        # Volatility filter: ATR > 20-period average
        vol_filter = atr_6h[i] > atr_ma_6h[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above 20-period high
        breakout_down = close[i] < low_20[i-1]  # Break below 20-period low
        
        if position == 0:
            # Long: Breakout up with uptrend and volatility
            if trend_up and vol_filter and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down with downtrend and volatility
            elif trend_down and vol_filter and breakout_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 20-period low or trend reversal
            if close[i] < low_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 20-period high or trend reversal
            if close[i] > high_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0