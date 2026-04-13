#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation
# Combines trend following (Alligator) with bull/bear power (Elder Ray) for robust signals
# Volume confirmation filters weak moves
# Designed for 12h timeframe to capture multi-day trends in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    # Calculate SMAs
    sma_jaw = np.full(n, np.nan)
    sma_teeth = np.full(n, np.nan)
    sma_lips = np.full(n, np.nan)
    
    for i in range(jaw_period-1, n):
        sma_jaw[i] = np.mean(close[i-jaw_period+1:i+1])
    for i in range(teeth_period-1, n):
        sma_teeth[i] = np.mean(close[i-teeth_period+1:i+1])
    for i in range(lips_period-1, n):
        sma_lips[i] = np.mean(close[i-lips_period+1:i+1])
    
    # Apply offsets (shift right)
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    for i in range(jaw_offset, n):
        jaw[i] = sma_jaw[i-jaw_offset]
    for i in range(teeth_offset, n):
        teeth[i] = sma_teeth[i-teeth_offset]
    for i in range(lips_offset, n):
        lips[i] = sma_lips[i-lips_offset]
    
    # Elder Ray (13-period EMA)
    ema_period = 13
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, n):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema
    bear_power = low - ema
    
    # Align daily trend filter (50-period SMA)
    close_1d = df_1d['close'].values
    sma_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        sma_1d[i] = np.mean(close_1d[i-49:i+1])
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(19, n):
        avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(sma_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Williams Alligator signals: lips above teeth above jaw = uptrend
        # lips below teeth below jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: bull power > 0 and rising, bear power < 0 and falling
        bull_strong = bull_power[i] > 0 and (i == 0 or bull_power[i] > bull_power[i-1])
        bear_strong = bear_power[i] < 0 and (i == 0 or bear_power[i] < bear_power[i-1])
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Alligator uptrend + Elder Ray bullish + volume + price above daily SMA
            if (alligator_long and 
                bull_strong and 
                volume_confirm and
                price > sma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Alligator downtrend + Elder Ray bearish + volume + price below daily SMA
            elif (alligator_short and 
                  bear_strong and 
                  volume_confirm and
                  price < sma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns down OR Elder Ray turns bearish OR volume drops
            if (not alligator_long or 
                not bull_strong or
                vol < 0.7 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns up OR Elder Ray turns bullish OR volume drops
            if (not alligator_short or 
                not bear_strong or
                vol < 0.7 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0