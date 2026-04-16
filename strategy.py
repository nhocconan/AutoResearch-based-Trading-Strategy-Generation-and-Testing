#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
# Long when Bull Power > 0 and Bear Power < 0 (bulls in control) with EMA50 uptrend and volume > 1.5x MA20.
# Short when Bear Power < 0 and Bull Power < 0 (bears in control) with EMA50 downtrend and volume > 1.5x MA20.
# Exit when Elder Power diverges (Bull Power <= 0 for long, Bear Power >= 0 for short) or opposite signal.
# Uses discrete position size 0.25. Elder Ray measures market power via EMA, effective in both bull/bear regimes.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing regime shifts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Elder Ray and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get weekly data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA50 for trend direction ===
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume moving average (20-period) on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        bp = bull_power_aligned[i]
        br = bear_power_aligned[i]
        ema50_val = ema50_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (bulls lose control) or Bear Power turns positive
            if bp <= 0 or br >= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns positive (bears lose control) or Bull Power turns positive
            if br >= 0 or bp >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: EMA50 slope (using 3-bar momentum for trend direction)
            if i >= 3:
                ema50_slope = ema50_val - ema50_aligned[i-3]
                uptrend = ema50_slope > 0
                downtrend = ema50_slope < 0
            else:
                uptrend = False
                downtrend = False
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Bull Power > 0 and Bear Power < 0 (bulls in control) with uptrend and volume
            if (bp > 0 and br < 0) and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 and Bull Power < 0 (bears in control) with downtrend and volume
            elif (br < 0 and bp < 0) and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dElderRay_1wEMA50_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0