#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray with 1d regime filter (ADX + chop).
- Primary timeframe: 6h targeting 80-120 total trades over 4 years (20-30/year).
- HTF: 1d for regime classification (ADX + chop) and Elder Ray calculation.
- Entry: Long when Alligator bullish (JAW > TEETH > LIPS) AND Bull Power > 0 AND regime = trending (ADX>25 & chop<61.8).
         Short when Alligator bearish (JAW < TEETH < LIPS) AND Bear Power < 0 AND regime = trending.
- Exit: Opposite Alligator alignment OR regime shifts to range (ADX<20 OR chop>61.8).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Alligator uses SMAs (13,8,5) with smoothing; Elder Ray uses 13-period EMA.
- Works in bull markets (catch trends) and bear markets (avoid whipsaws via regime filter).
- Estimated trades: ~100 total over 4 years (~25/year) based on trend persistence with regime confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    tr = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (tr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (tr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_vals = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx_vals

def choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr_sum = pd.Series(atr(high, low, close, 1)).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d regime indicators: ADX and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d = choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=1)
    
    # Calculate 1d Elder Ray and Alligator
    ema13_1d = ema(df_1d['close'].values, 13)
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = ema13_1d - df_1d['low'].values
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d, additional_delay_bars=1)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d, additional_delay_bars=1)
    
    # Calculate 1d Alligator (JAW=13, TEETH=8, LIPS=5)
    jaw_1d = sma(ema(df_1d['close'].values, 13), 13)
    teeth_1d = sma(ema(df_1d['close'].values, 8), 8)
    lips_1d = sma(ema(df_1d['close'].values, 5), 5)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d, additional_delay_bars=1)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d, additional_delay_bars=1)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: trending if ADX>25 AND chop<61.8, else range
        is_trending = adx_1d_aligned[i] > 25.0 and chop_1d_aligned[i] < 61.8
        
        # Exit conditions: opposite Alligator alignment OR regime shifts to range
        if position != 0:
            # Exit long: Alligator turns bearish OR regime becomes range
            if position == 1:
                if not (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]) or not is_trending:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR regime becomes range
            elif position == -1:
                if not (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i]) or not is_trending:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with Elder Ray confirmation in trending regime
        if position == 0 and is_trending:
            # Long: Alligator bullish AND Bull Power > 0
            if jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i] and bull_power_1d_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0
            elif jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i] and bear_power_1d_aligned[i] > 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Alligator_ElderRay_1dRegimeFilter_v1"
timeframe = "6h"
leverage = 1.0