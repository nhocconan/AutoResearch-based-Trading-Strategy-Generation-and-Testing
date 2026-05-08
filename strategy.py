#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 12h Supertrend trend filter with 4h RSI mean reversion
# and volume confirmation. Uses 12h Supertrend for trend direction (works in bull/bear via ATR),
# 4h RSI(14) for mean reversion entries, and volume spike confirmation.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing
# mean reversion within the trend context. Works in both bull (buy dips in uptrend) and
# bear (sell rallies in downtrend) markets.

name = "4h_Supertrend12h_RSI14_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Supertrend on 12h (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full_like(close_12h, np.nan, dtype=float)
    for i in range(atr_period, len(close_12h)):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend calculation
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    if not np.isnan(atr[atr_period]):
        supertrend[atr_period] = upper_band[atr_period]
        direction[atr_period] = 1
    
    for i in range(atr_period + 1, len(close_12h)):
        if np.isnan(supertrend[i-1]):
            # Continue initialization if needed
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                supertrend[i] = upper_band[i]
                direction[i] = 1
            continue
            
        if close_12h[i] <= supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = lower_band[i]
            direction[i] = 1
            
        # Adjust bands
        if direction[i] == 1:  # Uptrend
            if lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            supertrend[i] = lower_band[i]
        else:  # Downtrend
            if upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 4h timeframe
    direction_12h = direction
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # 4h RSI(14) for mean reversion
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    # Wilder's smoothing
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.nanmean(gain[1:15])
            avg_loss[i] = np.nanmean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14 + 10)  # Ensure enough data for RSI and Supertrend
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion entries in trend direction
            if direction_aligned[i] == 1:  # Uptrend - look for dips
                if rsi[i] < 35 and vol_confirm[i]:  # Oversold with volume
                    signals[i] = 0.25
                    position = 1
            elif direction_aligned[i] == -1:  # Downtrend - look for rallies
                if rsi[i] > 65 and vol_confirm[i]:  # Overbought with volume
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: RSI reverts to mean or trend changes
            if rsi[i] > 65 or direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI reverts to mean or trend changes
            if rsi[i] < 35 or direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals