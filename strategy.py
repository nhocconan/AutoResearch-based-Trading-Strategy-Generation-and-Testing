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
    
    # Get daily data for HTF context (trend and volatility)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA20 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h ATR for stop loss
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 0-23 UTC (12h bars cover full day, no session filter needed)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above average (avoid choppy markets)
        vol_filter = atr_12h[i] > atr_1d_aligned[i] * 0.8
        
        # Trend filter: price above/below daily EMA20
        trend_up = close[i] > ema20_1d_aligned[i]
        trend_down = close[i] < ema20_1d_aligned[i]
        
        # Entry conditions: 
        # Long: price above daily EMA20 with volume and volatility
        # Short: price below daily EMA20 with volume and volatility
        long_entry = trend_up and vol_filter
        short_entry = trend_down and vol_filter
        
        # Exit conditions: 
        # Long exit: price crosses below daily EMA20 OR ATR drops (range market)
        long_exit = not trend_up or (atr_12h[i] < atr_1d_aligned[i] * 0.5)
        # Short exit: price crosses above daily EMA20 OR ATR drops (range market)
        short_exit = not trend_down or (atr_12h[i] < atr_1d_aligned[i] * 0.5)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "12h_DailyEMA20_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0