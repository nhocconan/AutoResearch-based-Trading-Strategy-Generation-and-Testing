#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Regime
Hypothesis: On 1h timeframe, enter long when price breaks above daily Camarilla H3 level with volume confirmation and 4h uptrend (price > 4h EMA20). Enter short when price breaks below daily L3 level with volume confirmation and 4h downtrend (price < 4h EMA20). Use daily volatility regime filter: only trade when daily ATR(14) is above its 50-period median (avoid low-volatility chop). Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_Volume_Regime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: OHLC for Camarilla levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3/L3)
    daily_range = high_1d - low_1d
    camarilla_H3 = close_1d + 1.1 * daily_range / 4
    camarilla_L3 = close_1d - 1.1 * daily_range / 4
    
    # Align to 1h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # === DAILY VOLATILITY REGIME FILTER: ATR(14) > median of ATR(50) ===
    # Calculate ATR(14)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR(50) median (using 50-period window)
    atr_50_median = np.full_like(atr_14, np.nan)
    for i in range(50, len(atr_14)):
        atr_50_median[i] = np.median(atr_14[i-50:i])
    
    vol_regime = atr_14 > atr_50_median
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # === 4H INDICATORS: EMA(20) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume filter: volume > 1.5 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Breakout conditions with volume confirmation and volatility regime
        long_breakout = (close[i] > camarilla_H3_aligned[i]) and volume_filter[i] and vol_regime_aligned[i]
        short_breakout = (close[i] < camarilla_L3_aligned[i]) and volume_filter[i] and vol_regime_aligned[i]
        
        # Exit conditions: trend reversal or reversion to mean (price back inside H3/L3)
        exit_long = not uptrend or (close[i] < camarilla_L3_aligned[i])
        exit_short = not downtrend or (close[i] > camarilla_H3_aligned[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals