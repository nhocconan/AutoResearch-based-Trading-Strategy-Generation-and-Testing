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
    
    # Get 4h data for trend filter (12h HTF specified in prompt)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for higher timeframe trend (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate 12h EMA21 for higher timeframe trend
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate 4h ATR14 for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h_arr, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h_arr, 1))
    tr1[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(atr14_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: both 4h and 12h EMA must agree
        trend_up = close[i] > ema21_4h_aligned[i] and close[i] > ema21_12h_aligned[i]
        trend_down = close[i] < ema21_4h_aligned[i] and close[i] < ema21_12h_aligned[i]
        
        # Volatility filter: avoid extreme volatility
        vol_filter = atr14_4h_aligned[i] < (np.mean(atr14_4h_aligned[max(0, i-20):i+1]) * 1.5)
        
        # Entry conditions: price must be near EMA with momentum
        # Long: price above both EMAs with volume
        long_condition = trend_up and volume_filter[i] and vol_filter
        
        # Short: price below both EMAs with volume
        short_condition = trend_down and volume_filter[i] and vol_filter
        
        # Exit conditions: trend reversal or volatility expansion
        long_exit = (not trend_up) or (not vol_filter)
        short_exit = (not trend_down) or (not vol_filter)
        
        # Handle entries and exits
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_EMA21_DualTrend_VolumeFilter_12hHTF"
timeframe = "4h"
leverage = 1.0