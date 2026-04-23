#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 12h EMA50 AND volume > 1.5x 20-period average
- Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 12h EMA50 AND volume > 1.5x 20-period average
- Exit when Elder Ray momentum weakens (Bull Power < 0 for longs, Bear Power < 0 for shorts)
- Uses 12h EMA50 for trend alignment to avoid counter-trend trades and capture major trend
- Volume confirmation ensures institutional participation and reduces false breakouts
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Elder Ray Bull/Bear Power (13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 51, 20)  # Need 20 for volume MA, 51 for EMA50 (50+1), 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
        bearish_momentum = bear_power[i] > 0 and bull_power[i] < 0
        
        # Trend filter
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish momentum + uptrend + volume confirmation
            if bullish_momentum and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + downtrend + volume confirmation
            elif bearish_momentum and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray momentum weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: bullish momentum weakens (Bull Power < 0)
                if bull_power[i] < 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: bearish momentum weakens (Bear Power < 0)
                if bear_power[i] < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0