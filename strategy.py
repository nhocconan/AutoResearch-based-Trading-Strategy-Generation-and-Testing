#!/usr/bin/env python3
"""
12h_trix_volume_regime_v1
Hypothesis: On 12-hour timeframe, use TRIX momentum with volume confirmation and volatility regime filter.
Long when TRIX crosses above zero, 12h volume > 1.5x 20-period average, and 12h ADX < 25 (ranging market).
Short when TRIX crosses below zero, 12h volume > 1.5x 20-period average, and 12h ADX < 25.
Exit when TRIX crosses back through zero or ADX rises above 25 indicating trend.
Designed for 20-40 trades/year to minimize fee drag while capturing momentum reversals in ranging markets.
Works in both bull/bear markets as TRIX captures momentum shifts and volume filter ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_regime_v1"
timeframe = "12h"
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
    
    # Calculate TRIX on 12h data (1-period ROC of triple EMA)
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Calculate ADX for regime filter
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DM smoothed, -DM smoothed, TR smoothed over 14 periods
    # ADX = 100 * smoothed(|+DI - -DI| / (+DI + -DI))
    
    # Calculate directional movement
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    di_sum = plus_dm14 + minus_dm14
    dx = np.where(di_sum != 0, 100 * np.abs(plus_dm14 - minus_dm14) / di_sum, 0)
    adx = wilders_smoothing(dx, 14)
    adx = np.nan_to_num(adx, nan=0.0)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, 20), n):
        # Skip if data not available
        if (np.isnan(trix[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR ADX > 25 (trending)
            if trix[i] < 0 or adx[i] > 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR ADX > 25 (trending)
            if trix[i] > 0 or adx[i] > 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and ranging market (ADX < 25)
            if vol_ok and adx[i] < 25:
                # Long: TRIX crosses above zero
                if trix[i] > 0 and trix[i-1] <= 0:
                    position = 1
                    signals[i] = 0.25
                # Short: TRIX crosses below zero
                elif trix[i] < 0 and trix[i-1] >= 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals