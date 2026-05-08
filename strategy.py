#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Keltner Channel breakout with 1d ADX trend filter and volume confirmation
# Keltner Channel uses ATR-based bands around EMA. Breakout above upper band signals bullish momentum,
# breakdown below lower band signals bearish momentum. Confirmed by 1d ADX > 25 (trending market)
# and volume spike (>1.5x 20-period average). Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "12h_Keltner_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]  # first value
    low_close[0] = high_low[0]   # first value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DI and -DI
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate TR for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Keltner Channel on 12h data
    ema20 = ema(close, 20)
    atr10 = atr(high, low, close, 10)
    upper_keltner = ema20 + (2.0 * atr10)
    lower_keltner = ema20 - (2.0 * atr10)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(ema20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        upper_keltner_val = upper_keltner[i]
        lower_keltner_val = lower_keltner[i]
        ema20_val = ema20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close above upper Keltner + strong trend (ADX > 25) + volume spike
            if (close[i] > upper_keltner_val and 
                adx_val > 25 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Close below lower Keltner + strong trend (ADX > 25) + volume spike
            elif (close[i] < lower_keltner_val and 
                  adx_val > 25 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below EMA20 OR trend weakens (ADX < 20)
            if close[i] < ema20_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above EMA20 OR trend weakens (ADX < 20)
            if close[i] > ema20_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals