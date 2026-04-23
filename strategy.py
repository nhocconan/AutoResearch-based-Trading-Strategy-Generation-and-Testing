#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
Breakouts at R4 (long) and S4 (short) are rare but high-probability events when confirmed by
1d trend and volume spike. Uses 1d EMA50 for trend alignment to avoid counter-trend trades.
Volume > 2x average ensures conviction. Target: 50-150 total trades over 4 years (12-37/year).
Works in bull/bear markets by only taking trend-aligned breakouts.
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
    
    # Load 1d data for EMA50 trend filter and Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate previous day's Camarilla levels (R4, S4) - requires prior day's OHLC
    # We need to shift by 1 to use previous completed day's data
    if len(high_1d) < 2:
        return np.zeros(n)
    
    prev_high = high_1d[:-1]  # yesterday's high
    prev_low = low_1d[:-1]    # yesterday's low
    prev_close = close_1d[:-1] # yesterday's close
    
    # Calculate Camarilla levels for previous day
    rang = prev_high - prev_low
    # R4 = close + 1.5 * range
    r4 = prev_close + 1.5 * rang
    # S4 = close - 1.5 * range
    s4 = prev_close - 1.5 * rang
    
    # Align to 6h timeframe (these levels are valid for today's trading session)
    r4_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], r4)  # use df without last row
    s4_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], s4)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R4 AND price > 1d EMA50 AND volume spike
            if (price > r4_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND price < 1d EMA50 AND volume spike
            elif (price < s4_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse breakout or volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S4 OR volume drops below average
                if (price < s4_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R4 OR volume drops below average
                if (price > r4_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4S4_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0