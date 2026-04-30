#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-bar avg.
# Short: Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-bar avg.
# Exit: Opposite Elder Ray signal (Bull Power crosses below 0 for long exit, Bear Power crosses above 0 for short exit).
# Uses Elder Ray to measure power of bulls/bears relative to EMA13, with 1d EMA50 for higher-timeframe trend alignment.
# Volume confirmation filters low-conviction moves. Designed to work in both bull and bear markets by requiring trend alignment.
# Timeframe: 6h, HTF: 1d as per experiment guidelines.

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirmation_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) 
            #        AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND Bull Power < 0 (bulls weak)
            #        AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (curr_bear_power < 0 and 
                  curr_bull_power < 0 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Bull Power crosses below 0 (bulls losing control)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power crosses above 0 (bears losing control)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals