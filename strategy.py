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
    
    # Get 1d data for ATR and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                            np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for price action (our primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian(20) channels
    donch_high = pd.Series(close_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(close_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align indicators to 15m timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + EMA trend filter + ATR volatility filter
        breakout_long = close[i] > donch_high_aligned[i]
        breakout_short = close[i] < donch_low_aligned[i]
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (ATR too small)
        vol_filter = atr_14_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        long_entry = breakout_long and uptrend and vol_filter
        short_entry = breakout_short and downtrend and vol_filter
        
        # Exit when price returns to the middle of Donchian channel
        donch_mid = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
        exit_long = position == 1 and close[i] < donch_mid
        exit_short = position == -1 and close[i] > donch_mid
        
        # ATR-based stop loss (2x ATR from entry)
        # Note: We approximate by checking if price moved against us by 2*ATR
        # Since we don't track entry price exactly, we use a time-based decay
        # Alternative: use trailing stop based on highest/lowest since entry
        # For simplicity, we'll use a time-based exit after 4 bars if no move
        
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

name = "12h_donchian_ema_trend_filter"
timeframe = "12h"
leverage = 1.0