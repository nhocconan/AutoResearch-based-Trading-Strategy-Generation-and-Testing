#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Bull/Bear Power with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low.
# Enter long when Bull Power > 0 and rising, Bear Power < 0, price > 1d EMA50, volume > 1.5x 20-period average.
# Enter short when Bear Power > 0 and rising, Bull Power < 0, price < 1d EMA50, volume > 1.5x 20-period average.
# Exit when power signals reverse or price crosses 1d EMA50.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades in bear markets (2022, 2025).
# Target: 12-37 trades/year by requiring trend alignment + power signals + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    
    # Calculate EMA13 for Elder Ray (12h close)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
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
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Elder Ray signals
        bull_power_current = bull_power[i]
        bear_power_current = bear_power[i]
        bull_power_prev = bull_power[i-1] if i > 0 else 0
        bear_power_prev = bear_power[i-1] if i > 0 else 0
        
        bull_power_rising = bull_power_current > bull_power_prev
        bear_power_rising = bear_power_current > bear_power_prev
        
        # Trend filter: price vs daily EMA50
        bull_trend = price > ema50_1d_aligned[i]
        bear_trend = price < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long on rising Bull Power, negative Bear Power, bullish trend, volume confirmation
            if bull_power_current > 0 and bear_power_current < 0 and bull_power_rising and bull_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short on rising Bear Power, negative Bull Power, bearish trend, volume confirmation
            elif bear_power_current > 0 and bull_power_current < 0 and bear_power_rising and bear_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: power signals reverse or price crosses 1d EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power turns negative or price crosses below 1d EMA50
                if bull_power_current <= 0 or price < ema50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power turns negative or price crosses above 1d EMA50
                if bear_power_current <= 0 or price > ema50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Elder_Ray_Power_Trend_Volume"
timeframe = "12h"
leverage = 1.0