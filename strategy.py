#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h trend filter (EMA200) + volume confirmation
# Elder Ray: BullPower = High - EMA13, BearPower = Low - EMA13
# Go long when BullPower > 0 and rising + volume > 1.3x average + price > 12h EMA200
# Go short when BearPower < 0 and falling + volume > 1.3x average + price < 12h EMA200
# Exit when Elder Power reverses sign or volume drops
# Target: 12-37 trades/year by requiring trend alignment + Elder Ray divergence + volume filter
# Works in bull/bear: Elder Ray detects institutional buying/selling pressure, EMA200 filters trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA200 to 6h
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate Elder Ray components on 6h data
    ema13 = pd.Series(prices['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13  # High - EMA13
    bear_power = prices['low'].values - ema13   # Low - EMA13
    
    # Calculate volume average (20-period)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema200_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma
        
        # Trend filter: price vs 12h EMA200
        bull_trend = price > ema200_12h_aligned[i]
        bear_trend = price < ema200_12h_aligned[i]
        
        # Elder Ray signals with slope (need prior value)
        if i > 0:
            bull_rising = bull > bull_power[i-1]
            bear_falling = bear < bear_power[i-1]
        else:
            bull_rising = False
            bear_falling = False
        
        if position == 0:
            # Enter long: BullPower > 0 and rising + volume + bullish trend
            if bull > 0 and bull_rising and volume_confirm and bull_trend:
                signals[i] = 0.25
                position = 1
            # Enter short: BearPower < 0 and falling + volume + bearish trend
            elif bear < 0 and bear_falling and volume_confirm and bear_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Elder Power reverses or volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: BullPower <= 0 or volume confirmation lost
                if bull <= 0 or not volume_confirm:
                    exit_signal = True
            elif position == -1:
                # Exit short: BearPower >= 0 or volume confirmation lost
                if bear >= 0 or not volume_confirm:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_12hEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0