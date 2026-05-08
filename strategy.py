#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1-day EMA34 trend filter and volume spike.
# Uses proven price channel (Camarilla) breakouts with trend alignment and volume confirmation.
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h high-low range for Camarilla calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Camarilla levels (based on previous 12h candle)
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    R4 = typical_price + 1.1 * range_12h / 2
    R3 = typical_price + 1.1 * range_12h / 4
    R2 = typical_price + 1.1 * range_12h / 6
    R1 = typical_price + 1.1 * range_12h / 12
    S1 = typical_price - 1.1 * range_12h / 12
    S2 = typical_price - 1.1 * range_12h / 6
    S3 = typical_price - 1.1 * range_12h / 4
    S4 = typical_price - 1.1 * range_12h / 2
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 + uptrend (price > 12h EMA34) + volume spike
            long_cond = (close[i] > R1_aligned[i]) and \
                        (close[i] > ema_34_12h_aligned[i]) and \
                        volume_spike[i]
            # Short: break below S1 + downtrend (price < 12h EMA34) + volume spike
            short_cond = (close[i] < S1_aligned[i]) and \
                         (close[i] < ema_34_12h_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below S1 (mean reversion to support)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above R1 (mean reversion to resistance)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals