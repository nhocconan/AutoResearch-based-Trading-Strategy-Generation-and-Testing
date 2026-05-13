#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Alligator combination with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND Alligator jaws < teeth < lips (bullish alignment) AND close > 1d EMA50 AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND Alligator jaws > teeth > lips (bearish alignment) AND close < 1d EMA50 AND volume > 1.5x average
# Exit when Williams %R crosses -50 (mean reversion) OR Alligator alignment breaks OR trend reversal (price crosses 1d EMA50)
# Uses 6h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with daily trend filter for BTC/ETH resilience.
# Williams %R identifies extreme momentum exhaustion, Alligator confirms trend direction, volume validates breakout authenticity.

name = "6h_WilliamsR_Alligator_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h data (primary timeframe)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate Alligator on 6h data (primary timeframe)
    # Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift Alligator lines (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted values that don't have enough data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 1.5x 20-period average (volume confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Check Alligator alignment
        bullish_alignment = jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]
        bearish_alignment = jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND bullish Alligator alignment AND close > 1d EMA50 AND volume spike
            if williams_r[i] < -80 and bullish_alignment and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND bearish Alligator alignment AND close < 1d EMA50 AND volume spike
            elif williams_r[i] > -20 and bearish_alignment and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion) OR Alligator alignment breaks OR trend reversal
            if williams_r[i] > -50 or not bullish_alignment or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion) OR Alligator alignment breaks OR trend reversal
            if williams_r[i] < -50 or not bearish_alignment or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals