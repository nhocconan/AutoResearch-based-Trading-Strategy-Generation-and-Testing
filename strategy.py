#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ATR regime filter and volume spike confirmation
# In low volatility regimes (1d ATR < 20-period SMA): breakout continuation at R4/S4 levels
# In high volatility regimes (1d ATR >= 20-period SMA): mean reversion at R3/S3 levels
# Volume confirmation (>2.0x 20-period EMA) filters low-quality signals
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
# Strategy adapts to changing volatility regimes and avoids whipsaw in choppy markets.

name = "6h_Camarilla_R3S3_R4S4_1dATR_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = smoothed TR (using Wilder's smoothing: alpha = 1/period)
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR 20-period SMA for regime filter
    atr_sma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe (completed 1d bar only)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_sma_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20)
    
    # Calculate 1d Camarilla levels (R3, S3, R4, S4) from previous day
    # Previous day's OHLC (1d data shifted by 1)
    prev_close_1d = close_1d.shift(1)
    prev_high_1d = high_1d.shift(1)
    prev_low_1d = low_1d.shift(1)
    prev_range = prev_high_1d - prev_low_1d
    
    # Camarilla levels using previous day's data
    camarilla_r4 = prev_close_1d + 1.5 * prev_range
    camarilla_r3 = prev_close_1d + 1.0 * prev_range
    camarilla_s3 = prev_close_1d - 1.0 * prev_range
    camarilla_s4 = prev_close_1d - 1.5 * prev_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_aligned[i]) or np.isnan(atr_sma_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        # Regime determination: low volatility if ATR < SMA, high volatility otherwise
        low_vol_regime = atr_aligned[i] < atr_sma_aligned[i]
        
        if position == 0:
            if low_vol_regime:
                # Low volatility: breakout continuation at R4/S4
                if close[i] > r4_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s4_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # High volatility: mean reversion at R3/S3
                if close[i] <= s3_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r3_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to midpoint between R3 and S3 OR volatility regime shifts to high vol
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] >= midpoint or 
                not low_vol_regime):  # Exit if regime shifts to high volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR volatility regime shifts to high vol
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if (close[i] <= midpoint or 
                not low_vol_regime):  # Exit if regime shifts to high volatility
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals