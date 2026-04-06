#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 12h EMA50 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 12h EMA50 AND volume > 1.5x average
# Exit when Elder Power signals reverse or price crosses 12h EMA50
# Uses 6h timeframe for balance of signal frequency and noise reduction
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA13 of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    twelve_hour_close = df_12h['close'].values
    twelve_hour_close_s = pd.Series(twelve_hour_close)
    twelve_hour_ema = twelve_hour_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    twelve_hour_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_hour_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(twelve_hour_ema_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit if Bull Power turns negative OR price crosses below 12h EMA
            if bull_power[i] <= 0 or close[i] < twelve_hour_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if Bear Power turns positive OR price crosses above 12h EMA
            if bear_power[i] >= 0 or close[i] > twelve_hour_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power positive AND Bear Power rising (less negative) AND price > 12h EMA AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                i > 0 and bear_power[i] > bear_power[i-1] and  # Bear Power rising
                close[i] > twelve_hour_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND Bull Power falling (less positive) AND price < 12h EMA AND volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  i > 0 and bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  close[i] < twelve_hour_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals