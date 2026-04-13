#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d ATR(14) volatility filter and 1d close > SMA(50) trend filter
    # Long when: price breaks above Donchian(20) high AND 1d close > SMA(50) (bull regime) AND ATR(14) > 1.2x 20-bar avg ATR
    # Short when: price breaks below Donchian(20) low AND 1d close < SMA(50) (bear regime) AND ATR(14) > 1.2x 20-bar avg ATR
    # Exit when: price crosses Donchian(20) midpoint OR 1d close crosses SMA(50) in opposite direction
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Volatility filter ensures breakouts occur during expansion, reducing false signals in chop.
    # Trend filter prevents counter-trend trades in strong regimes.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d SMA(50) for trend filter
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 4h ATR(14) for volatility confirmation
    tr1_4h = pd.Series(high - low)
    tr2_4h = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3_4h = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    avg_atr_4h = pd.Series(atr_14_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(avg_atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period low
        
        # 1d trend filter: close > SMA(50) for uptrend, < SMA(50) for downtrend
        uptrend = close_1d[-1] > sma_50_1d_aligned[i] if len(close_1d) > 0 else False  # Use last known 1d close
        downtrend = close_1d[-1] < sma_50_1d_aligned[i] if len(close_1d) > 0 else False
        
        # Volatility confirmation: 4h ATR > 1.2x 20-bar avg ATR (breakout during expansion)
        vol_expansion = atr_14_4h[i] > (1.2 * avg_atr_4h[i])
        
        # Entry conditions
        long_entry = breakout_up and uptrend and vol_expansion and position != 1
        short_entry = breakout_down and downtrend and vol_expansion and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_mid[i] or close_1d[-1] < sma_50_1d_aligned[i]))
        exit_short = (position == -1 and (close[i] > donchian_mid[i] or close_1d[-1] > sma_50_1d_aligned[i]))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_1d_donchian_atr_sma_v1"
timeframe = "4h"
leverage = 1.0