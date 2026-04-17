#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above Camarilla R1 level AND volume > 1.5x average AND 12h EMA34 > previous EMA34.
Short when price breaks below Camarilla S1 level AND volume > 1.5x average AND 12h EMA34 < previous EMA34.
Exit when price reverts to Camarilla midpoint (PP) or opposite S1/R1 level is touched.
Uses 4h for price action and volume, 12h for EMA trend filter to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday
support/resistance, volume confirms breakout strength, 12h EMA ensures alignment with higher timeframe trend.
Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).
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
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe (based on previous 4h bar)
    # Camarilla: PP = (H+L+C)/3, R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    # We use the previous completed 4h bar's HLC to calculate levels for current bar
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + 1.1 * (high_4h - low_4h) / 12.0
    s1_4h = close_4h - 1.1 * (high_4h - low_4h) / 12.0
    
    # Shift by 1 to use previous bar's levels (no look-ahead)
    pp_4h = np.roll(pp_4h, 1)
    r1_4h = np.roll(r1_4h, 1)
    s1_4h = np.roll(s1_4h, 1)
    pp_4h[0] = r1_4h[0] = s1_4h[0] = np.nan  # first value invalid after roll
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h timeframe
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h Camarilla levels to 4h timeframe (no alignment needed for same TF)
    pp_aligned = pp_4h
    r1_aligned = r1_4h
    s1_aligned = s1_4h
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Calculate EMA34 slope (current vs previous) for trend filter
        if i > 0 and not np.isnan(ema_34_aligned[i-1]):
            ema_34_slope = ema_34_val - ema_34_aligned[i-1]
        else:
            ema_34_slope = 0
        
        if position == 0:
            # Long: price > R1 AND volume > 1.5x avg AND EMA34 rising (bullish trend)
            if price > r1_val and vol > 1.5 * vol_ma and ema_34_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.5x avg AND EMA34 falling (bearish trend)
            elif price < s1_val and vol > 1.5 * vol_ma and ema_34_slope < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP (reversion to mean) OR price touches S1 (strong reversal)
            if price < pp_val or price < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP (reversion to mean) OR price touches R1 (strong reversal)
            if price > pp_val or price > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0