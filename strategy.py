#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA(close,34), Bear Power = EMA(close,34) - Low.
# Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND 1d EMA34 up AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND Bull Power < previous Bull Power (bearish momentum) AND 1d EMA34 down AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Elder Ray measures power of bulls/bears relative to trend EMA.
# 1d EMA34 ensures we only trade with higher timeframe trend (avoiding counter-trend whipsaws).
# Volume spike confirms institutional participation. Designed to capture strong trending moves in both bull and bear markets.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Components ===
    # EMA34 of close for Bull/Bear Power calculation
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    bull_power = high - ema34  # High - EMA34
    bear_power = ema34 - low   # EMA34 - Low
    
    # Previous period values for momentum check
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34+1 periods needed for EMA34 and momentum, 20 for volume MA)
    warmup = 55
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(bull_power_prev[i]) or
            np.isnan(bear_power_prev[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        bull_prev = bull_power_prev[i]
        bear_prev = bear_power_prev[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative OR Bear Power becomes positive (momentum shift)
            if bull_val <= 0 or bear_val >= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns negative OR Bull Power becomes positive (momentum shift)
            if bear_val <= 0 or bull_val >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND 1d EMA34 up AND volume spike
            if bull_val > 0 and bear_val < bear_prev and ema34_1d_val > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power > 0 AND Bull Power < previous Bull Power (bearish momentum) AND 1d EMA34 down AND volume spike
            elif bear_val > 0 and bull_val < bull_prev and ema34_1d_val < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0