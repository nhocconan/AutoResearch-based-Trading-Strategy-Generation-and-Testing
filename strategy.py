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
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channel (25) on 1d for stability
    donchian_high = pd.Series(high_1d).rolling(window=25, min_periods=25).max().values
    donchian_low = pd.Series(low_1d).rolling(window=25, min_periods=25).min().values
    
    # EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR20 on 1d for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Volume confirmation: current volume > 1.8x 25-period average
    vol_ma_25 = pd.Series(volume).rolling(window=25, min_periods=25).mean().values
    volume_surge = volume > (vol_ma_25 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_20_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_20_aligned[i] > 0.008 * close[i]  # ATR > 0.8% of price
        
        # Entry conditions
        # Long: upward breakout + uptrend + volume surge + vol filter
        long_entry = breakout_up and trend_up and volume_surge[i] and vol_filter
        # Short: downward breakout + downtrend + volume surge + vol filter
        short_entry = breakout_down and trend_down and volume_surge[i] and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.28
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.28
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.28  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.28   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.28
            elif position == -1:
                signals[i] = -0.28
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian25_Breakout_1dEMA50_Volume_VolFilter"
timeframe = "12h"
leverage = 1.0