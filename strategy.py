#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Elder Ray + weekly trend + volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long: Bull Power > 0 and Bear Power < 0 (bullish momentum) + price above weekly EMA26 + volume > 20-period average
# Short: Bear Power < 0 and Bull Power > 0 (bearish momentum) + price below weekly EMA26 + volume > 20-period average
# Uses Elder Ray for momentum, weekly trend for direction, volume for confirmation.
# Designed for ~15-30 trades/year per symbol.
name = "6h_ElderRay_WeeklyTrend_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_26 = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_1w_26_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_26)
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_26_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_weekly = ema_1w_26_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bullish momentum (Bull Power > 0, Bear Power < 0) + above weekly EMA26 + volume confirmation
            if bull > 0 and bear < 0 and price > ema_weekly and vol_current > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish momentum (Bear Power < 0, Bull Power > 0) + below weekly EMA26 + volume confirmation
            elif bear < 0 and bull > 0 and price < ema_weekly and vol_current > vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Momentum turns bearish or price breaks below weekly EMA
            if bull <= 0 or bear >= 0 or price < ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Momentum turns bullish or price breaks above weekly EMA
            if bear >= 0 or bull <= 0 or price > ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals