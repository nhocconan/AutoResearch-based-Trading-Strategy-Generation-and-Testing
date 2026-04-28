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
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channel (10) on 12h
    donchian_high = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    # EMA20 on 12h for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR10 on 12h for volatility filter
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    atr_10_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    # Volume confirmation: current volume > 1.8x 10-period average
    vol_ma_10 = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_10)
    volume_surge = volume > (vol_ma_10_aligned * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_10_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 12h EMA20
        trend_up = close[i] > ema_20_12h_aligned[i]
        trend_down = close[i] < ema_20_12h_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_10_aligned[i] > 0.008 * close[i]  # ATR > 0.8% of price
        
        # Entry conditions
        # Long: upward breakout + uptrend + volume surge + vol filter
        long_entry = breakout_up and trend_up and volume_surge[i] and vol_filter
        # Short: downward breakout + downtrend + volume surge + vol filter
        short_entry = breakout_down and trend_down and volume_surge[i] and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian10_Breakout_12hEMA20_Volume_VolFilter"
timeframe = "6h"
leverage = 1.0