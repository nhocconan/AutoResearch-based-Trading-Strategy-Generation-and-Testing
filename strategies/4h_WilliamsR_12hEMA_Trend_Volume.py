#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R combined with 12h EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions (below -80 for long, above -20 for short).
# 12h EMA provides trend direction: only take longs when price > 12h EMA, shorts when price < 12h EMA.
# Volume confirmation requires current volume > 1.5x 20-period average to avoid false signals.
# Designed to work in both bull and bear markets by aligning with trend via 12h EMA filter.
# Targets 20-40 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h data
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Williams %R on 4h data (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    wr = np.where(denominator != 0, -100 * ((highest_high - close) / denominator), -50)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(wr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr_val = wr[i]
        ema_val = ema_12h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: oversold + uptrend + volume spike
            if wr_val < -80 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought + downtrend + volume spike
            elif wr_val > -20 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R returns to overbought or trend breaks
                if wr_val > -20 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R returns to oversold or trend breaks
                if wr_val < -80 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0