#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with volume confirmation and weekly EMA20 filter.
Long when price breaks above R1 with volume > 1.5x 20-period average AND close > weekly EMA20.
Short when price breaks below S1 with volume > 1.5x 20-period average AND close < weekly EMA20.
Exit when price returns to the Camarilla pivot (PP) or volume drops below average.
Designed for low trade frequency (7-25/year) on 1d timeframe to minimize fee drag and work in both bull/bear regimes.
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
    
    # Get 1d data for indicator calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (using previous day's OHLC)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We shift by 1 to avoid look-ahead (use previous day's data)
    pp = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r1 = np.roll(close_1d, 1) + (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12
    s1 = np.roll(close_1d, 1) - (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12
    
    # Calculate weekly EMA20 on 1d timeframe (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema20_1w = ema20_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND close > weekly EMA20
            if price > r1_val and vol > 1.5 * vol_ma and price > ema20_1w:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation AND close < weekly EMA20
            elif price < s1_val and vol > 1.5 * vol_ma and price < ema20_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot (PP) OR volume drops below average
            if price <= pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot (PP) OR volume drops below average
            if price >= pp_val or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Volume_1wEMA20_Filter"
timeframe = "1d"
leverage = 1.0