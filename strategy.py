#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme with 4h volume confirmation and ATR-based stoploss.
# Long when 1d Williams %R < -80 (oversold) AND 4h volume > 1.5x 20-period average AND price > 4h EMA(50).
# Short when 1d Williams %R > -20 (overbought) AND 4h volume > 1.5x 20-period average AND price < 4h EMA(50).
# Exit when price crosses the 4h EMA(50) in the opposite direction.
# Uses discrete position size 0.25. 1d Williams %R provides mean-reversion edge in ranging markets,
# volume confirms participation, 4h EMA(50) filters trend alignment and provides dynamic stop.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for EMA and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: EMA(50) ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: Williams %R (14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_14_1d = (highest_high_14_1d - close_1d) / (highest_high_14_1d - lowest_low_14_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_14_1d = np.where((highest_high_14_1d - lowest_low_14_1d) == 0, -50, williams_r_14_1d)
    
    # Align all indicators to primary timeframe (4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14_1d)
    
    # Calculate 4h volume moving average (no alignment needed as it's LTF)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        ema_50 = ema_50_aligned[i]
        williams_r = williams_r_aligned[i]
        vol_ma_20 = vol_ma_20_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < ema_50:  # Exit when price crosses below EMA(50)
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > ema_50:  # Exit when price crosses above EMA(50)
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND volume > 1.5x 20-period avg AND price > EMA(50)
            if (williams_r < -80) and (vol > 1.5 * vol_ma_20) and (price > ema_50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND volume > 1.5x 20-period avg AND price < EMA(50)
            elif (williams_r > -20) and (vol > 1.5 * vol_ma_20) and (price < ema_50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dWilliamsR_Extreme_Volume_EMA50Filter_V1"
timeframe = "4h"
leverage = 1.0