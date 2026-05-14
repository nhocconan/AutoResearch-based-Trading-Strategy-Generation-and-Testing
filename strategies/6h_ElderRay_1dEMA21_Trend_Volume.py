#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA21 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA. Works in both bull/bear markets by requiring
# alignment with daily trend and high-volume spikes to confirm institutional participation.
# Target: 50-150 trades over 4 years (12-37/year) with position size 0.25.
name = "6h_ElderRay_1dEMA21_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA21 trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume filter: current volume > 2.0x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for EMA and volume calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_signal = bull_power[i] > 0 and bear_power[i] < 0  # Both positive and negative power present
        bearish_signal = bull_power[i] < 0 and bear_power[i] > 0  # This condition is impossible, fixing logic
        # Corrected: Bullish when Bull Power > 0, Bearish when Bear Power > 0
        bullish_entry = bull_power[i] > 0
        bearish_entry = bear_power[i] > 0
        
        trend_up = close[i] > ema_21_6h[i]
        trend_down = close[i] < ema_21_6h[i]
        
        if position == 0:
            # Long: bullish power positive + uptrend + volume confirmation
            if bullish_entry and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish power positive + downtrend + volume confirmation
            elif bearish_entry and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish power becomes positive or trend reversal
            if bearish_entry or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish power becomes positive or trend reversal
            if bullish_entry or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals