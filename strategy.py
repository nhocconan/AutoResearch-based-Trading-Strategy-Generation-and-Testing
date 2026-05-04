#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Long when Lips > Teeth > Jaw and price above Alligator, Short when Lips < Teeth < Jaw and price below Alligator
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.5x 20 EMA volume) filters false signals
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets (trend continuation) and bear markets (trend continuation at lower levels)
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter from prior completed 1d bar
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_shifted = np.roll(ema_50_1d, 1)
    ema_50_1d_shifted[0] = np.nan
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams Alligator components (13, 8, 5 periods) from prior completed 12h bar
    # Jaw: 13-period SMMA shifted by 8 bars
    # Teeth: 8-period SMMA shifted by 5 bars
    # Lips: 5-period SMMA shifted by 3 bars
    # SMMA (Smoothed Moving Average) calculation: first value is SMA, subsequent values are smoothed
    
    def smma(values, period):
        """Calculate Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple moving average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: (prev_smma * (period-1) + current_value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    # Shift to align with Alligator strategy (Jaw shifted by 8, Teeth by 5, Lips by 3)
    jaw = np.roll(smma_13, 8)
    teeth = np.roll(smma_8, 5)
    lips = np.roll(smma_5, 3)
    
    # Set initial values to NaN to avoid look-ahead
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > lips[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < lips[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Lips OR Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if close[i] < lips[i] or lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Lips OR Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if close[i] > lips[i] or lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals