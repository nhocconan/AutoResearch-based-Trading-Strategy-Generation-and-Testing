#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Mean Reversion with 1d EMA50 trend filter and volume spike confirmation.
- Long when Williams %R(14) crosses above -80 (oversold reversal) AND close > 1d EMA50 (bullish trend) AND volume > 2.0 * volume SMA(20)
- Short when Williams %R(14) crosses below -20 (overbought reversal) AND close < 1d EMA50 (bearish trend) AND volume > 2.0 * volume SMA(20)
- Exit on Williams %R crossing above -20 (long) or below -80 (short) or trend reversal
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Williams %R captures mean reversion extremes that work in both bull (buy dips) and bear (sell rallies) markets
- 1d EMA50 ensures alignment with longer-term trend to avoid counter-trend whipsaws
- Volume spike filter (2.0x average) confirms institutional participation, reducing false signals
- Designed for BTC/ETH with edge in ranging markets via mean reversion and in trending markets via trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) using previous period (no look-ahead)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20) for volume spike confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (2.0 * volume_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal), trend up (close > EMA50), volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal), trend down (close < EMA50), volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR trend reverses
            if williams_r[i] > -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR trend reverses
            if williams_r[i] < -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0