#!/usr/bin/env python3
"""
6h_WilliamsAlligator_Trend_With_Volume_Filter_v1
Hypothesis: 6h Williams Alligator (Smoothed MA crossover) with 1d trend filter and volume confirmation.
- Long when Alligator jaws < teeth < lips (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5 * volume_ma(20)
- Short when Alligator jaws > teeth > lips (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5 * volume_ma(20)
- Uses Williams Alligator from 6h chart for trend identification
- 1d EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume confirmation (1.5x) reduces false signals
- Designed for moderate frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Exit on opposite Alligator alignment or 1d trend reversal
- Novelty: Williams Alligator is under-explored in crypto; combines trend, HTF filter, and volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter (needs completed daily candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Williams Alligator on 6h chart (primary timeframe)
    # Alligator: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift for Alligator's forward projection (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for daily EMA, 13 for Alligator jaw, 20 for volume MA)
    start_idx = max(50, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Alligator alignment conditions with trend and volume spike filter
        if position == 0:
            # Long: Jaw < Teeth < Lips (bullish) AND 1d uptrend AND volume spike
            if jaw[i] < teeth[i] < lips[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish) AND 1d downtrend AND volume spike
            elif jaw[i] > teeth[i] > lips[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bearish Alligator alignment OR 1d trend turns down
            if jaw[i] > teeth[i] > lips[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bullish Alligator alignment OR 1d trend turns up
            if jaw[i] < teeth[i] < lips[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsAlligator_Trend_With_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0