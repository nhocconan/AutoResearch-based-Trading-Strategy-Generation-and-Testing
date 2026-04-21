#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: -100 * (HH - Close) / (HH - LL)
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close_1d) / (highest_high - lowest_low),
        -50  # neutral when range is zero
    )
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to lower timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: Williams %R oversold + RSI rising from oversold + volume surge
            if (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and
                rsi_aligned[i] > 30 and rsi_aligned[i-1] <= 30 and
                vol_1d_current > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought + RSI falling from overbought + volume surge
            elif (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and
                  rsi_aligned[i] < 70 and rsi_aligned[i-1] >= 70 and
                  vol_1d_current > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range or volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns above -50 or volume < average
                if (williams_r_aligned[i] > -50 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns below -50 or volume < average
                if (williams_r_aligned[i] < -50 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_RSI_Volume2x"
timeframe = "1d"
leverage = 1.0