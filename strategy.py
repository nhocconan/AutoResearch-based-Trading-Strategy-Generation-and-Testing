#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
# Long when Bull Power > 0 AND price > 1w EMA50 AND 6h volume > 1.3x 20-period average.
# Short when Bear Power < 0 AND price < 1w EMA50 AND 6h volume > 1.3x 20-period average.
# Exit when power crosses zero or price crosses EMA50.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength via EMA13.
# Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 6h data once before loop for volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # === 1d Indicators: Elder Ray (Bull/Bear Power) ===
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # === 1w Indicators: EMA50 ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h Indicators: Volume average ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume aligned
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        vol_6h_current = vol_6h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if bull_power <= 0 or price < ema50:  # Exit when bull power fades or price breaks EMA50 down
                exit_signal = True
        
        elif position == -1:  # Short position
            if bear_power >= 0 or price > ema50:  # Exit when bear power fades or price breaks EMA50 up
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND price > 1w EMA50 AND 6h volume > 1.3x 20-period avg
            if (bull_power > 0) and (price > ema50) and (vol_6h_current > 1.3 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND price < 1w EMA50 AND 6h volume > 1.3x 20-period avg
            elif (bear_power < 0) and (price < ema50) and (vol_6h_current > 1.3 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dElderRay_Power_1wEMA50_TrendFilter_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0