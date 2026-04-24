#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
- Williams Alligator (Jaw/Teeth/Lips) defines trend direction: price > Teeth = uptrend, price < Teeth = downtrend.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low. Enter long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
- 1d EMA50 trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
- Volume confirmation: current volume > 1.5x 20-bar average to ensure conviction.
- Discrete position size 0.25 to manage drawdown and reduce fee churn.
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient.
- Works in bull/bear: Alligator avoids whipsaws in ranging markets, Elder Ray measures power behind moves, 1d trend ensures higher timeframe alignment.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (completed 1d bar)
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price > Teeth (uptrend) AND Bull Power > 0 AND Bull Power rising (vs prev) AND price > 1d EMA50 AND volume confirmation
            if (close[i] > teeth[i] and 
                bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price < Teeth (downtrend) AND Bear Power > 0 AND Bear Power rising (vs prev) AND price < 1d EMA50 AND volume confirmation
            elif (close[i] < teeth[i] and 
                  bear_power[i] > 0 and 
                  bear_power[i] > bear_power[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < Lips (weakening trend) OR Bull Power <= 0 (power fading)
            if close[i] < lips[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > Lips (weakening trend) OR Bear Power <= 0 (power fading)
            if close[i] > lips[i] or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0