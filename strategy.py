#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme readings with 1d EMA34 trend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND price > 6h open AND 1d EMA34 is rising AND volume > 1.5 * avg_volume(20) on 6h
# Short when 1d Williams %R > -20 (overbought) AND price < 6h open AND 1d EMA34 is falling AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses the 6h EMA20 (dynamic stop/reversal)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R captures exhaustion moves in both bull and bear markets
# EMA34 ensures we trade with the daily trend while reducing noise
# Volume confirmation filters out low-conviction moves
# 6h EMA20 exit provides adaptive risk control

name = "6h_1dWilliamsR_Extreme_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R and EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)  # avoid division by zero
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA20 for exit
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), price above open (bullish intraday), EMA34 rising, volume spike
            if (williams_r_1d_aligned[i] < -80 and 
                close[i] > open_price[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below open (bearish intraday), EMA34 falling, volume spike
            elif (williams_r_1d_aligned[i] > -20 and 
                  close[i] < open_price[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 6h EMA20
            if close[i] < ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 6h EMA20
            if close[i] > ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals