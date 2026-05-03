#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) with volume > 2.0x 20-bar average and close > 1w EMA50 (uptrend)
# Short when Jaw > Teeth > Lips (bearish alignment) with volume > 2.0x 20-bar average and close < 1w EMA50 (downtrend)
# Exit when alignment breaks (Jaw-Teeth-Lips not in order)
# Williams Alligator identifies trend absence/presence via smoothed medians; works in trending markets with filters.
# Target: 30-100 total trades over 4 years = 7-25/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "1d_WilliamsAlligator_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (13,8,5) smoothed medians
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smoothed_mma(arr, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(arr, np.nan)
        smma[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            if not np.isnan(sma[i]) and not np.isnan(smma[i-1]):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
        return smma
    
    jaw = smoothed_mma(close, 13)
    teeth = smoothed_mma(close, 8)
    lips = smoothed_mma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 8, 5, 20) + 8  # SMMA periods + jaw shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Jaw < Teeth < Lips (bullish alignment) with volume spike and close > 1w EMA50 (uptrend)
            if (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Jaw > Teeth > Lips (bearish alignment) with volume spike and close < 1w EMA50 (downtrend)
            elif (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bullish alignment broken (Jaw >= Teeth or Teeth >= Lips)
            if jaw_shifted[i] >= teeth_shifted[i] or teeth_shifted[i] >= lips_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bearish alignment broken (Jaw <= Teeth or Teeth <= Lips)
            if jaw_shifted[i] <= teeth_shifted[i] or teeth_shifted[i] <= lips_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals