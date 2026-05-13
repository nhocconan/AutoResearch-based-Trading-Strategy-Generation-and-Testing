#!/usr/bin/env python3
"""
1d_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (15-period) captures momentum turning points; volume spike confirms strength; 
Choppiness Index regime filter (CHOP > 61.8 for range, < 38.2 for trend) selects appropriate strategy.
In ranging markets (CHOP > 61.8), fade TRIX extremes; in trending markets (CHOP < 38.2), follow TRIX crosses.
Uses weekly trend filter (price > weekly EMA40) for bias. Designed for low trade frequency (~10-25/year).
"""

name = "1d_TRIX_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX: triple-smoothed EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 15), 15), 15) * 100
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100
    
    # Volume spike: volume > 2.0 * 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR,14) / (max(high,14)-min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop[0:13] = np.nan
    chop_range = chop > 61.8  # ranging market
    chop_trend = chop < 38.2  # trending market
    
    # Weekly trend filter: price > weekly EMA40 for bullish bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    ema_40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    uptrend_1w = df_1w['close'].values > ema_40_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any key value is NaN
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        trix_val = trix[i]
        vol_spike_now = vol_spike[i]
        in_range = chop_range[i]
        in_trend = chop_trend[i]
        weekly_up = uptrend_1w_aligned[i]
        
        if position == 0:
            # LONG conditions
            if in_range:
                # In range: fade TRIX extremes (oversold bounce)
                if trix_val < -0.15 and vol_spike_now and weekly_up:
                    signals[i] = 0.25
                    position = 1
            else:
                # In trend: follow TRIX crosses (bullish momentum)
                if trix_val > 0.05 and vol_spike_now and weekly_up:
                    signals[i] = 0.25
                    position = 1
            # SHORT conditions
            if in_range:
                # In range: fade TRIX extremes (overbought pullback)
                if trix_val > 0.15 and vol_spike_now and not weekly_up:
                    signals[i] = -0.25
                    position = -1
            else:
                # In trend: follow TRIX crosses (bearish momentum)
                if trix_val < -0.05 and vol_spike_now and not weekly_up:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or volatility dies
            if trix_val < -0.05 or not vol_spike_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or volatility dies
            if trix_val > 0.05 or not vol_spike_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals