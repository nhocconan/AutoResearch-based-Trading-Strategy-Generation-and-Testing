#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend filter + volume confirmation
# Uses 6h timeframe for signal generation with Elder Ray oscillator
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long: Bull Power > 0 and increasing + price > 1d EMA34 + volume spike
# Short: Bear Power < 0 and decreasing + price < 1d EMA34 + volume spike
# Works in bull markets via trend-aligned strength, in bear via identifying weakness

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    # EMA13 for Bull/Bear Power calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Slope of Bull/Bear Power (1-period change) to detect increasing/decreasing
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    # Set first value to 0 (no previous bar)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND increasing + price > 1d EMA34 + volume confirm
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND decreasing + price < 1d EMA34 + volume confirm
            elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or Bear Power >= 0 (loss of bullish momentum)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or Bull Power <= 0 (loss of bearish momentum)
            if bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals