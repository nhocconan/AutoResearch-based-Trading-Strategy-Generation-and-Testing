#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter (EMA34) and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) + price > daily EMA34 + volume > 1.5x 20-period average
# Short when Bull Power < 0 and Bear Power > 0 (both bearish) + price < daily EMA34 + volume > 1.5x 20-period average
# Exit when Elder Ray signal weakens (Bull Power <= 0 for long, Bear Power <= 0 for short)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_Energy_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period average volume for normalization
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Elder Ray (Bull/Bear Power) on 6h timeframe
    # EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Elder Ray alignment + trend + volume
            long_condition = bull_power[i] > 0 and bear_power[i] > 0 and close[i] > ema_34_aligned[i] and vol_filter
            short_condition = bull_power[i] < 0 and bear_power[i] < 0 and close[i] < ema_34_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 (loss of bearish momentum)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals