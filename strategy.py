#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Chaikin Oscillator (CO) with 1d EMA50 trend filter and volume confirmation.
# Chaikin Oscillator = EMA3(ADL) - EMA10(ADL), where ADL = Accumulation/Distribution Line.
# CO > 0 indicates buying pressure, CO < 0 indicates selling pressure.
# 1d EMA50 ensures we trade in the direction of the daily trend (long when price > EMA50, short when price < EMA50).
# Volume confirmation: current volume > 1.5x 20-period average volume.
# This combination captures momentum with trend alignment, reducing false signals.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_ChaikinOscillator_1dEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Accumulation/Distribution Line (ADL)
    # Avoid division by zero: if high == low, use 0 (no money flow)
    hl_range = high - low
    clv = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0)
    adl = np.cumsum(clv * volume)
    
    # Calculate Chaikin Oscillator: EMA3(ADL) - EMA10(ADL)
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_adl - ema10_adl
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chaikin_osc[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Chaikin Oscillator > 0 (buying pressure) AND uptrend AND volume confirmation
            if chaikin_osc[i] > 0 and uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Chaikin Oscillator < 0 (selling pressure) AND downtrend AND volume confirmation
            elif chaikin_osc[i] < 0 and downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Chaikin Oscillator turns negative OR trend reverses
            if chaikin_osc[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Chaikin Oscillator turns positive OR trend reverses
            if chaikin_osc[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals