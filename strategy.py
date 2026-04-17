#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above R1 AND volume > 1.5x 20-period average AND price > 12h EMA34 (bullish trend).
Short when price breaks below S1 AND volume > 1.5x 20-period average AND price < 12h EMA34 (bearish trend).
Exit when price reverts to the Camarilla pivot point (PP).
Uses 4h for price/volume/Camarilla, 12h for EMA34 trend filter to avoid whipsaw in ranging markets.
Targets 75-200 total trades over 4 years (19-50/year). Camarilla levels provide high-probability reversal/breakout points,
volume confirmation reduces fakeouts, 12h EMA ensures we trade with the higher timeframe trend.
Works in bull markets (captures uptrends with bullish 12h EMA) and bear markets (captures downtrends with bearish 12h EMA).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 1-period Camarilla levels (R1, S1, PP) on 4h
    # R1 = Close + 1.1*(High-Low)/2
    # S1 = Close - 1.1*(High-Low)/2
    # PP = (High + Low + Close)/3
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 2
    camarilla_pp = (high_4h + low_4h + close_4h) / 3
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume_4h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h Camarilla levels, volume MA, and 12h EMA34 to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pp = camarilla_pp_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_34 = ema_34_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.5x avg AND price > 12h EMA34 (bullish trend)
            if price > r1 and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.5x avg AND price < 12h EMA34 (bearish trend)
            elif price < s1 and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP (pivot point)
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP (pivot point)
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA34_Filter"
timeframe = "4h"
leverage = 1.0