#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 1d Trend Filter and Volume Spike
# Elder Ray = Bull Power (High - EMA) + Bear Power (Low - EMA)
# Long when Bull Power > 0 and Bear Power rising + price > 1d EMA34 + volume spike
# Short when Bear Power < 0 and Bull Power falling + price < 1d EMA34 + volume spike
# Exit when power signals reverse or trend fails
# Combines trend strength with momentum and volume for low-frequency, high-quality signals.
# Designed for ~15-30 trades/year to minimize fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray Index on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 13-period EMA for Elder Ray (standard setting)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # 5-period smoothing for power signals (reduces noise)
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power rising (less negative) + uptrend + volume spike
            bear_rising = i > 50 and bear_val > bear_power_smooth[i-1]
            if bull_val > 0.0 and bear_rising and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power falling (less positive) + downtrend + volume spike
            elif bear_val < 0.0 and i > 50 and bull_val < bull_power_smooth[i-1] and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Power signals reverse or trend fails
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power turns negative or Bear Power rises above zero or trend fails
                if bull_val <= 0.0 or bear_val >= 0.0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power turns positive or Bull Power falls below zero or trend fails
                if bear_val >= 0.0 or bull_val <= 0.0 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ElderRay_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0