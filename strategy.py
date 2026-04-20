#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14) for oversold/overbought signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d RSI (14) for momentum confirmation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d ADX (14) for trend strength filter
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM-
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus14 / np.where(tr14 == 0, 1e-10, tr14)
    di_minus = 100 * dm_minus14 / np.where(tr14 == 0, 1e-10, tr14)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        williams_val = williams_r_aligned[i]
        rsi_val = rsi_aligned[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(williams_val) or np.isnan(rsi_val) or 
            np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + RSI > 50 + ADX > 25 (trending)
            if williams_val < -80 and rsi_val > 50 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + RSI < 50 + ADX > 25 (trending)
            elif williams_val > -20 and rsi_val < 50 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or RSI < 40
            if williams_val > -20 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or RSI > 60
            if williams_val < -80 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_WilliamsRSI_ADXFilter_V1
# Uses 1-day Williams %R for overbought/oversold signals
# Enters long when Williams %R < -80 (oversold) + RSI > 50 + ADX > 25
# Enters short when Williams %R > -20 (overbought) + RSI < 50 + ADX > 25
# Uses ADX > 25 to ensure trending market (avoid chop)
# Exits on opposite Williams %R level or RSI extreme
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_WilliamsRSI_ADXFilter_V1"
timeframe = "12h"
leverage = 1.0