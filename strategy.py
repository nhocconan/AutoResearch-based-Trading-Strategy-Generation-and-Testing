#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h EMA filter and volume confirmation.
# Uses 4h EMA50 to determine trend direction and 1h EMA20 for entry timing.
# Only trades in direction of 4h trend during high-liquidity session (08-20 UTC).
# Volume filter requires 1.5x average volume to avoid false breakouts.
# Designed for low trade frequency (15-35/year) to minimize fee drag in ranging markets.
# Works in both bull and bear markets by following established 4h trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h EMA20 for entry timing
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period average volume for volume filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema4h = ema_4h_aligned[i]
        ema20 = ema_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5 * 20-period EMA volume
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above both EMAs and volume filter
            if price > ema20 and price > ema4h and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below both EMAs and volume filter
            elif price < ema20 and price < ema4h and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: price crosses back below/above EMA20
            exit_signal = False
            
            if position == 1:  # long position
                if price < ema20:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > ema20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA20_EMA50_Trend_Filter_Volume"
timeframe = "1h"
leverage = 1.0