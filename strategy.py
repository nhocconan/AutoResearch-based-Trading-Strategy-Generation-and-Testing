#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate Donchian channels (20-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-period high
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Average True Range for stop loss (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above 20-day Donchian upper band + volume confirmation
        price_above_upper = price_close > upper_20_aligned[i]
        if price_above_upper and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below 20-day Donchian lower band + volume confirmation
        price_below_lower = price_close < lower_20_aligned[i]
        if price_below_lower and vol_confirm:
            enter_short = True
        
        # Exit conditions: ATR-based stop loss and profit target
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Stop loss: 2.5 * ATR below entry
            # Take profit: 3.0 * ATR above entry (trailing not simulated)
            # Using close-based exit only
            if price_close < upper_20_aligned[i] - 0.5 * (upper_20_aligned[i] - lower_20_aligned[i]):
                exit_long = True
        elif position == -1:
            if price_close > lower_20_aligned[i] + 0.5 * (upper_20_aligned[i] - lower_20_aligned[i]):
                exit_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Donchian breakout with daily channel and volume confirmation.
# Uses 20-day Donchian channels from daily timeframe for structural breakouts.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Works in both bull and bear markets by capturing breakouts from key daily levels.
# Position size 0.25 to manage drawdown in volatile markets.
# Target: 20-40 trades/year to minimize fee drag.