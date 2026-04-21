#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Breakout_Volume_Regime
Hypothesis: 1h Camarilla (R1/S1) breakout filtered by 4h EMA50 trend and 1d volume regime.
In trending markets (price > EMA50_4h): breakout continuation (long above R1, short below S1).
In ranging markets (1d chop > 61.8): no entries to avoid whipsaw. Uses volume confirmation (1.8x average) to filter false breakouts.
ATR(14) stoploss (2.0x) and discrete position sizing (0.20) to limit fee drag and drawdown.
Designed to work in both bull and bear markets by requiring strong trend alignment and avoiding choppy regimes.
Timeframe: 1h, uses 4h for trend and 1d for regime filter.
Target: 60-150 total trades over 4 years = 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA50 trend, 1d for chop regime)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d OHLC for Choppiness Index regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).sum().values  # Sum for CHOP denominator
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (ATR(14) * 14)) / log10(14)
    # We need ATR(14) of true range for numerator
    atr_1d_val = tr_1d.rolling(window=14, min_periods=14).mean().values
    chop_denominator = atr_1d_val * 14
    chop_ratio = np.where((atr_1d > 0) & (chop_denominator > 0), atr_1d / chop_denominator, 1.0)
    chop_ratio = np.clip(chop_ratio, 0.001, 1000.0)  # Avoid log(0)
    chop_1d = 100 * (np.log10(chop_ratio) / np.log10(14))
    
    # Align 1d Choppiness Index to 1h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 0.275 * range_1d
    s1_1d = close_1d - 0.275 * range_1d
    h3_1d = close_1d + 1.1 * range_1d
    l3_1d = close_1d - 1.1 * range_1d
    h4_1d = close_1d + 1.382 * range_1d
    l4_1d = close_1d - 1.382 * range_1d
    
    # Align 1d Camarilla levels to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])
            or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        h4 = h4_1d_aligned[i]
        l4 = l4_1d_aligned[i]
        ema_trend = ema_50_4h_aligned[i]
        chop = chop_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        regime_filter = chop < 61.8
        
        if position == 0:
            # Only enter in trending markets AND non-choppy regime
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > r1) and (price > ema_trend) and volume_confirmed and regime_filter
            short_condition = (price < s1) and (price < ema_trend) and volume_confirmed and regime_filter
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at H4 (extreme overbought)
            elif price > h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at L4 (extreme oversold)
            elif price < l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Camarilla_Breakout_Volume_Regime"
timeframe = "1h"
leverage = 1.0