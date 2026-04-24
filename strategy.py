#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND close > 1d EMA50 (bullish trend)
- Short when price breaks below Camarilla L3 AND close < 1d EMA50 (bearish trend)
- Volume must be > 1.8 * median volume of last 20 bars (volume spike filter)
- Exit on opposite Camarilla breakout or trend reversal (close crosses 1d EMA50)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Camarilla H3/L3 levels provide stronger breakout signals than R1/S1 with fewer false triggers
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws in both bull and bear markets
- Volume spike filter adapts to changing market conditions, reducing fakeouts
- Designed for BTC/ETH edge in ranging markets (mean reversion at extremes) and trending markets (breakout continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous bar's range)
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.8 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.8 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA50), volume spike
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3, trend down (close < EMA50), volume spike
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla L3 OR trend reversal (close < EMA50)
            if close[i] < camarilla_l3[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla H3 OR trend reversal (close > EMA50)
            if close[i] > camarilla_h3[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0