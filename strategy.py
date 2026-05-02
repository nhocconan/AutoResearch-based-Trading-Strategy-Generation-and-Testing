#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets: Bull Power > 0 and rising = strong uptrend, Bear Power < 0 and falling = strong downtrend
# In ranging markets: Oscillations around zero provide mean reversion signals at extremes
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike (1.8x 20-period average) confirms conviction in the move
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend)
# Targets 50-150 trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_BullBearPower_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate volume spike (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA13, volume MA and HTF data alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 and rising (bullish momentum) + price > 1d EMA50 + volume spike
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling (bearish momentum) + price < 1d EMA50 + volume spike
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive (loss of bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals