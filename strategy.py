#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Williams Alligator + Volume + Chop Filter
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Combined with volume confirmation to avoid false breakouts and Choppiness Index
# to filter ranging markets, this strategy captures trending moves in both bull
# and bear markets. The Alligator's convergence/divergence provides clear signals
# while minimizing whipsaws. Target: 15-25 trades/year to minimize fee drag.
name = "1d_williams_alligator_volume_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period, shift):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        # Apply shift (displacement)
        if shift > 0:
            result = np.roll(result, shift)
            result[:shift] = np.nan
        return result
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) - ranging market filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Weekly trend filter (1w) - using EMA(21) for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i]) or np.isnan(weekly_ema_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Lips cross below Teeth (Alligator waking up to eat) OR weekly trend turns bearish
            if lips[i] < teeth[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Lips cross above Teeth OR weekly trend turns bullish
            if lips[i] > teeth[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in trending markets (CHOP < 61.8) with volume confirmation
            if chop[i] < 61.8 and volume[i] > vol_ma[i]:
                # Enter long: Lips > Teeth > Jaw (Alligator eating with mouth open up) AND bullish weekly trend
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > weekly_ema_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: Lips < Teeth < Jaw (Alligator eating with mouth open down) AND bearish weekly trend
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < weekly_ema_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals