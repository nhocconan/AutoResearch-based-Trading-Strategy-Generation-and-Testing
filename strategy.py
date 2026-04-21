#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, price > 1d EMA34, volume > 1.5x average.
# Short when Bear Power < 0 and falling, price < 1d EMA34, volume > 1.5x average.
# Uses EMA13 for power calculation (standard) and EMA34 on 1d for trend filter.
# Target: 15-35 trades/year by requiring EMA13 crossover, trend alignment, and volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray power (13-period)
    close = prices['close']
    ema13 = close.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = (prices['high'] - ema13).values
    bear_power = (prices['low'] - ema13).values
    
    # Calculate EMA of Bull/Bear Power to detect rising/falling (13-period)
    bull_power_ema = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_ema = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Load 1d EMA34 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_ema[i]) or np.isnan(bear_power_ema[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = close.iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Elder Ray conditions
        bull_power_rising = bull_power[i] > bull_power_ema[i]
        bear_power_falling = bear_power[i] < bear_power_ema[i]
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Trend filter from 1d EMA34
        uptrend = price > ema34_1d_aligned[i]
        downtrend = price < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive and rising, uptrend, volume confirmation
            if bull_power_positive and bull_power_rising and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative and falling, downtrend, volume confirmation
            elif bear_power_negative and bear_power_falling and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power becomes negative or turns down
                if bull_power[i] <= 0 or not bull_power_rising:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power becomes positive or turns up
                if bear_power[i] >= 0 or not bear_power_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0