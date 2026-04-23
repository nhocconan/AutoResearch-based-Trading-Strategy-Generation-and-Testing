#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above 12h Camarilla R1 level AND price > 1d EMA34 (uptrend) AND volume > 1.5x average.
Short when price breaks below 12h Camarilla S1 level AND price < 1d EMA34 (downtrend) AND volume > 1.5x average.
Exit when price reverts to 12h Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA34).
Uses 12h timeframe with tight entry conditions (Camarilla R1/S1 are strong resistance/support) to limit trades to 12-37/year.
1d EMA34 provides smooth trend filter. Volume spike ensures high-conviction breakouts.
Target: 50-150 trades over 4 years (12-37/year) as proven by DB for 12h timeframe.
Works in both bull (trend following) and bear (mean reversion at pivot) markets via dual exit logic.
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
    
    # Calculate 12h Camarilla levels (R1, S1, PP) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels on 12h (based on previous 12h bar's OHLC)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_pp = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    camarilla_r1 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 4.0
    camarilla_s1 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 4.0
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R1 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r1_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S1 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s1_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 1d EMA34 (trend reversal)
                if price <= pp_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 1d EMA34 (trend reversal)
                if price >= pp_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0