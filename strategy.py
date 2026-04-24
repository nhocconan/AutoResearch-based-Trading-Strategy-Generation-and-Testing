#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla regime filter: Use 1d choppiness index (CHOP) to avoid ranging markets.
- Entry: Long when price breaks above Donchian(20) upper band AND 1d EMA50 bullish AND volume > 1.5 * volume MA(20) AND CHOP < 61.8 (trending).
         Short when price breaks below Donchian(20) lower band AND 1d EMA50 bearish AND volume > 1.5 * volume MA(20) AND CHOP < 61.8 (trending).
- Exit: Close-based reversal - exit long when price crosses below Donchian(20) middle (10-period average),
        exit short when price crosses above Donchian(20) middle.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Stoploss: Implicit via exit on middle band cross (no separate stop needed).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(df_1d['high'].values - df_1d['low'].values,
                    np.maximum(np.abs(df_1d['high'].values - np.append(df_1d['close'].values[0], df_1d['close'].values[:-1])),
                               np.abs(df_1d['low'].values - np.append(df_1d['close'].values[0], df_1d['close'].values[:-1]))))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_atr14) / np.log10(14) / np.log10((hh14 - ll14) + 1e-10)
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)  # fill NaN with neutral value
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate Donchian channels on 4h data (20-period)
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    # Donchian middle = (upper + lower) / 2
    dh20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dl20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dm20 = (dh20 + dl20) / 2.0
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(dh20[i]) or np.isnan(dl20[i]) or np.isnan(dm20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and trending regime (CHOP < 61.8)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            trending_regime = chop_aligned[i] < 61.8
            
            # Long: Price breaks above Donchian upper AND 1d EMA50 bullish AND volume confirmed AND trending
            if curr_high > dh20[i] and curr_close > ema_1d_aligned[i] and vol_confirmed and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND 1d EMA50 bearish AND volume confirmed AND trending
            elif curr_low < dl20[i] and curr_close < ema_1d_aligned[i] and vol_confirmed and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian middle (trend weakening)
            if curr_close < dm20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian middle (trend weakening)
            if curr_close > dm20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0