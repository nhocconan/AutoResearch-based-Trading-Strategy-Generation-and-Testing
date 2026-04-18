#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + 1d EMA50 Trend Filter
# Williams Alligator uses smoothed SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# When all three lines are aligned (JAW > TEETH > LIPS for uptrend, reverse for downtrend),
# it indicates strong trend direction. Combined with volume spike for confirmation
# and 1d EMA50 for higher timeframe trend alignment, this filters false signals.
# Works in bull markets (buy when aligned above EMA50) and bear markets (sell when aligned below EMA50).
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drift.
name = "12h_WilliamsAlligator_Volume_1dEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d data for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate volume spike: current volume > 2.0 * 10-period average volume
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up (JAW > TEETH > LIPS) AND price above EMA50 AND volume spike
            if jaw_val > teeth_val and teeth_val > lips_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down (JAW < TEETH < LIPS) AND price below EMA50 AND volume spike
            elif jaw_val < teeth_val and teeth_val < lips_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks (JAW <= TEETH or TEETH <= LIPS) or price below EMA50
            if jaw_val <= teeth_val or teeth_val <= lips_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks (JAW >= TEETH or TEETH >= LIPS) or price above EMA50
            if jaw_val >= teeth_val or teeth_val >= lips_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals