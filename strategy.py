#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike with 1d trend filter.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction and alignment
- Elder Ray (Bull/Bear Power with 13-period EMA) for momentum confirmation
- Volume > 1.8x 20-period average for spike confirmation
- 1d EMA50 as higher timeframe trend filter
- Position size: 0.25 discrete level to minimize fee churn
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Works in bull/bear via Alligator alignment + Elder Ray momentum + volume confirmation
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: Smoothed Moving Average (SMA) with specific periods
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50)  # Volume MA, Alligator jaws, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        alligator_bull = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bear = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        if position == 0:
            # Long: Alligator bullish alignment AND Elder Ray bull power positive AND volume confirmation AND price above 1d EMA50
            if (alligator_bull and elder_bull and volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND Elder Ray bear power negative AND volume confirmation AND price below 1d EMA50
            elif (alligator_bear and elder_bear and volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment OR Elder Ray turns negative OR price crosses below 1d EMA50
            if (not alligator_bull or not elder_bull or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment OR Elder Ray turns positive OR price crosses above 1d EMA50
            if (not alligator_bear or not elder_bear or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_1dEMA50_v1"
timeframe = "4h"
leverage = 1.0