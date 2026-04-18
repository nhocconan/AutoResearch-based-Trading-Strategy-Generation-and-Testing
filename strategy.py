#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Chandelier Exit trailing stop with 1d Supertrend for direction.
# Chandelier Exit adapts to volatility (ATR-based) to avoid whipsaw in chop, Supertrend filters trend direction.
# Works in bull (follows trend) and bear (avoids false breakouts via volatility filter).
# Target: 15-40 trades/year (60-160 total over 4 years) to minimize fee drag.
name = "12h_ChandelierExit_Supertrend_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Supertrend and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(hl2)
    direction = np.ones_like(hl2)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2)):
        if np.isnan(atr[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            supertrend[i] = hl2[i]
            direction[i] = direction[i-1]
            continue
            
        if close[i-1] > supertrend[i-1]:
            # Previous close was above previous supertrend -> uptrend
            upper_band[i] = min(upper_band[i], upper_band[i-1])
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            # Previous close was below previous supertrend -> downtrend
            lower_band[i] = max(lower_band[i], lower_band[i-1])
            upper_band[i] = min(upper_band[i], upper_band[i-1])
            
        if close[i] > upper_band[i]:
            direction[i] = 1
        elif close[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align daily Supertrend and ATR to 12h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate Chandelier Exit for 12h (22, 3.0) - trailing stop based on ATR
    chandelier_period = 22
    chandelier_multiplier = 3.0
    
    # True Range for 12h
    tr1_12h = np.abs(high - low)
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]  # First value
    atr_12h = pd.Series(tr_12h).rolling(window=chandelier_period, min_periods=chandelier_period).mean().values
    
    # Chandelier Exit: for long positions, subtract ATR*multiplier from highest high
    # For short positions, add ATR*multiplier to lowest low
    highest_high = pd.Series(high).rolling(window=chandelier_period, min_periods=chandelier_period).max().values
    lowest_low = pd.Series(low).rolling(window=chandelier_period, min_periods=chandelier_period).min().values
    
    long_exit = highest_high - chandelier_multiplier * atr_12h
    short_exit = lowest_low + chandelier_multiplier * atr_12h
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, chandelier_period)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(long_exit[i]) or np.isnan(short_exit[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        supertrend_val = supertrend_aligned[i]
        long_exit_val = long_exit[i]
        short_exit_val = short_exit[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: Supertrend uptrend, price above Chandelier long exit, volume confirmation
            if supertrend_val < close_val and close_val > long_exit_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, price below Chandelier short exit, volume confirmation
            elif supertrend_val > close_val and close_val < short_exit_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below Chandelier long exit or Supertrend turns down
            if close_val < long_exit_val or supertrend_val > close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above Chandelier short exit or Supertrend turns up
            if close_val > short_exit_val or supertrend_val < close_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals