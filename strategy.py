#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray combo with volume confirmation and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend context (optional filter).
- Entry: Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x MA20 volume.
         Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x MA20 volume.
- Exit: Opposite Alligator alignment OR Elder Ray power crosses zero.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator: SMAs of median price (13,8,5) with offsets (8,5,3).
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Chop regime: Only trade when Chop(14) < 61.8 (avoid strong ranging markets).
- Works in bull markets (buy alignments in uptrend) and bear markets (sell alignments in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on strict alignment requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def median_price(high, low):
    """Calculate median price."""
    return (high + low) / 2.0

def bull_power(high, ema_close):
    """Calculate Elder Ray Bull Power."""
    return high - ema_close

def bear_power(low, ema_close):
    """Calculate Elder Ray Bear Power."""
    return low - ema_close

def chop(high, low, close, period):
    """Calculate Choppiness Index."""
    atr_sum = pd.Series(np.abs(high - low)).rolling(window=period, min_periods=period).sum()
    hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop_val = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop_val.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Williams Alligator components on 12h
    med_price = median_price(high, low)
    jaws = sma(med_price, 13)  # 13-period
    teeth = sma(med_price, 8)   # 8-period
    lips = sma(med_price, 5)    # 5-period
    
    # Apply Alligator offsets (8,5,3)
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray components
    ema13_close = ema(close, 13)
    bull = bull_power(high, ema13_close)
    bear = bear_power(low, ema13_close)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma20 = sma(volume, 20)
    vol_ratio = volume / (vol_ma20 + 1e-10)
    
    # Chop regime filter: avoid strong ranging markets
    chop_val = chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull[i]) or np.isnan(bear[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(chop_val[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR Elder Ray power crosses zero
        if position != 0:
            # Exit long: Alligator loses bullish alignment OR bull power <= 0
            if position == 1:
                if not (jaws[i] < teeth[i] and teeth[i] < lips[i]) or bull[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator loses bearish alignment OR bear power >= 0
            elif position == -1:
                if not (jaws[i] > teeth[i] and teeth[i] > lips[i]) or bear[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment + Elder Ray confirmation + volume + chop filter
        if position == 0:
            # Bullish alignment: jaws < teeth < lips
            bullish_align = jaws[i] < teeth[i] and teeth[i] < lips[i]
            # Bearish alignment: jaws > teeth > lips
            bearish_align = jaws[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long: bullish alignment AND bull power > 0 AND volume confirmation AND chop < 61.8 (not ranging)
            if (bullish_align and bull[i] > 0 and vol_ratio[i] > 1.5 and chop_val[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND bear power < 0 AND volume confirmation AND chop < 61.8
            elif (bearish_align and bear[i] < 0 and vol_ratio[i] > 1.5 and chop_val[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0