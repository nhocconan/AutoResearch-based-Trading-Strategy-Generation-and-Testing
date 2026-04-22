#!/usr/bin/env python3

"""
Hypothesis: 1-hour trend-following strategy using 4-hour Supertrend for direction and 1-hour price action for entry timing.
Trades in direction of 4-hour Supertrend, entering on 1-hour pullbacks to EMA21 with volume confirmation.
Uses session filter (08-20 UTC) to avoid low-liquidity periods. Designed for moderate trade frequency
(15-35 trades/year) to balance opportunity with fee minimization. Works in bull markets via trend continuation
and in bear markets via short-side participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final Bands
    final_upper = np.full_like(upper_band, np.nan)
    final_lower = np.full_like(lower_band, np.nan)
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    
    for i in range(1, len(close)):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
            
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # Supertrend
    supertrend = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_upper[i]
        elif supertrend[i-1] == final_upper[i-1]:
            if close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
        else:
            if close[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
    
    # Direction: 1 for uptrend, -1 for downtrend
    direction = np.where(close > supertrend, 1, -1)
    return direction, supertrend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Supertrend for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_dir, supertrend = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, supertrend_dir)
    
    # 1h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_ok:
            # Long: 4h uptrend + price pulls back to EMA21 + bounces
            if (supertrend_dir_aligned[i] == 1 and 
                close[i] > ema21[i] and 
                close[i-1] <= ema21[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price pulls back to EMA21 + rejects
            elif (supertrend_dir_aligned[i] == -1 and 
                  close[i] < ema21[i] and 
                  close[i-1] >= ema21[i-1]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: trend reversal or price moves against EMA significantly
            exit_signal = False
            
            if position == 1:
                # Exit long: 4h trend turns down or price breaks below EMA21
                if supertrend_dir_aligned[i] == -1 or close[i] < ema21[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: 4h trend turns up or price breaks above EMA21
                if supertrend_dir_aligned[i] == 1 or close[i] > ema21[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Supertrend4h_EMA21_Pullback_Volume"
timeframe = "1h"
leverage = 1.0