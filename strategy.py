#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and 1d volatility regime filter.
# Uses 4h Supertrend for trend direction, 1d ATR-based volatility regime (high/low vol),
# and enters on 1h pullbacks to EMA21 in the direction of the 4h trend.
# Only trades during high volatility regimes (ATR > 1.5x 20-day average) to avoid chop.
# Session filter: 08-20 UTC to avoid low-volume Asian session.
# Target: 80-150 total trades over 4 years (20-38/year) to stay within optimal range.

name = "1h_supertrend4h_volregime_ema21_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Supertrend for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for 4h
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_4h = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    
    # Supertrend calculation
    hl2_4h = (high_4h + low_4h) / 2
    upperband = hl2_4h + (multiplier * atr_4h)
    lowerband = hl2_4h - (multiplier * atr_4h)
    
    upperband_final = np.full_like(upperband, np.nan)
    lowerband_final = np.full_like(lowerband, np.nan)
    supertrend = np.full_like(close_4h, np.nan)
    trend = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] > upperband_final[i-1]:
            trend[i] = 1
        elif close_4h[i-1] < lowerband_final[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            upperband_final[i] = max(upperband[i], upperband_final[i-1])
            lowerband_final[i] = lowerband[i]
        else:
            upperband_final[i] = upperband[i]
            lowerband_final[i] = min(lowerband[i], lowerband_final[i-1])
        
        if trend[i] == 1 and close_4h[i] <= upperband_final[i]:
            supertrend[i] = upperband_final[i]
        elif trend[i] == -1 and close_4h[i] >= lowerband_final[i]:
            supertrend[i] = lowerband_final[i]
        elif trend[i] == 1:
            supertrend[i] = lowerband_final[i]
        else:
            supertrend[i] = upperband_final[i]
    
    # Determine trend direction: 1 for uptrend, -1 for downtrend
    trend_4h = np.where(close_4h > supertrend, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d ATR-based volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False).mean().values
    
    # 20-day average ATR for regime classification
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    high_vol_regime = atr_1d > (atr_ma_20 * 1.5)  # High volatility when ATR > 1.5x 20-day avg
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # 1h EMA21 for entry timing
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if any required data not available
        if (np.isnan(trend_4h_aligned[i]) or 
            np.isnan(high_vol_regime_aligned[i]) or 
            np.isnan(ema21[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check if we're in session and high volatility regime
        if not (in_session[i] and high_vol_regime_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: close crosses EMA21 against trend
        if position == 1:  # long position
            if close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: pullback to EMA21 in direction of 4h trend
            if trend_4h_aligned[i] == 1:  # 4h uptrend
                # Look for pullback to EMA21 from above
                if low[i] <= ema21[i] <= high[i]:
                    signals[i] = 0.20
                    position = 1
            elif trend_4h_aligned[i] == -1:  # 4h downtrend
                # Look for pullback to EMA21 from below
                if low[i] <= ema21[i] <= high[i]:
                    signals[i] = -0.20
                    position = -1
    
    return signals