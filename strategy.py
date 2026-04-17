#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.5x average AND 12h EMA34 > 12h EMA89 (uptrend).
Short when price breaks below Camarilla S1 AND volume > 1.5x average AND 12h EMA34 < 12h EMA89 (downtrend).
Exit when price reverts to Camarilla midpoint (PP) OR 12h EMA flattens (|EMA34-EMA89| < 0.1% of price).
Uses 4h for price/volume, 12h for EMA trend filter to avoid whipsaw in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide intraday support/resistance,
volume confirmation reduces fakeouts, 12h EMA ensures we only trade with intermediate-term trend.
Works in bull markets (captures uptrend continuations) and bear markets (captures downtrend continuations).
"""

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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels on 4h timeframe (using previous day's OHLC)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # We use the previous 4h bar's data to avoid look-ahead
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12.0
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (already on 4h, but need to shift for completed bar)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume_4h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp = pp_4h_aligned[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema34 = ema34_12h_aligned[i]
        ema89 = ema89_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.5x avg AND 12h EMA34 > EMA89 (uptrend)
            if price > r1 and vol > 1.5 * vol_ma and ema34 > ema89:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.5x avg AND 12h EMA34 < EMA89 (downtrend)
            elif price < s1 and vol > 1.5 * vol_ma and ema34 < ema89:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR |EMA34-EMA89| < 0.1% of price (trend weakening)
            if price < pp or np.abs(ema34 - ema89) < 0.001 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR |EMA34-EMA89| < 0.1% of price (trend weakening)
            if price > pp or np.abs(ema34 - ema89) < 0.001 * price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA_Trend_Filter"
timeframe = "4h"
leverage = 1.0