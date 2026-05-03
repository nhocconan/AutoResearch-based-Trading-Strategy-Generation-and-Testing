#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with 1d volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence via smoothed medians.
# ADX > 25 confirms trend strength. Enter long when price > LIPS and JAW > TEETH > LIPS (bullish alignment)
# with volume confirmation. Enter short when price < LIPS and JAW < TEETH < LIPS (bearish alignment).
# Uses 1d volume spike filter to avoid low-momentum breakouts. Designed for low trade frequency
# (12-37/year) on 6h timeframe to minimize fee drag. Works in bull markets via trend continuation
# and in bear markets via short signals during downtrends.

name = "6h_ADX_WilliamsAlligator_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (volume > 1.8 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.8 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Williams Alligator: 3 smoothed medians (SMMA)
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars
    # LIPS: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(np.median([high, low], axis=0), 13)  # Median price
    teeth = smma(np.median([high, low], axis=0), 8)
    lips = smma(np.median([high, low], axis=0), 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First TR is undefined
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan, dtype=np.float64)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(arr[:period])
            # Subsequent values: Wilder smoothing = prev * (1-1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
            return result
        
        atr = WilderSmoothing(tr, period)
        plus_di = 100 * WilderSmoothing(plus_dm, period) / atr
        minus_di = 100 * WilderSmoothing(minus_dm, period) / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = WilderSmoothing(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: JAW > TEETH > LIPS
            bullish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            # Bearish Alligator alignment: JAW < TEETH < LIPS
            bearish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            
            # Long: price > LIPS, bullish alignment, ADX > 25, volume spike
            if (close[i] > lips[i] and bullish_alignment and adx[i] > 25 and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < LIPS, bearish alignment, ADX > 25, volume spike
            elif (close[i] < lips[i] and bearish_alignment and adx[i] > 25 and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below LIPS or Alligator loses bullish alignment
            if close[i] < lips[i] or not (jaw[i] > teeth[i] and teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above LIPS or Alligator loses bearish alignment
            if close[i] > lips[i] or not (jaw[i] < teeth[i] and teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals