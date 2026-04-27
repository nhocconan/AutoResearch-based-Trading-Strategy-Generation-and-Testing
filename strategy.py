#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate daily RSI for overbought/oversold conditions
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1d_aligned[i]) or 
            np.isnan(lower_20_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to daily Donchian channels
        price_above_upper = close[i] > upper_20_1d_aligned[i]
        price_below_lower = close[i] < lower_20_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Volume filter: current volume above daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Long conditions: break above upper Donchian + RSI not overbought + volume
        long_condition = (price_above_upper and 
                         rsi_not_overbought and 
                         volume_filter)
        
        # Short conditions: break below lower Donchian + RSI not oversold + volume
        short_condition = (price_below_lower and 
                          rsi_not_oversold and 
                          volume_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: reversal signal
        elif position == 1 and price_below_lower:
            signals[i] = 0.0
            position = 0
        elif position == -1 and price_above_upper:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_1dRSI_VolumeFilter"
timeframe = "6h"
leverage = 1.0