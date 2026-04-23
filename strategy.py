#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot (R1/S1) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND price > 1d EMA34 AND volume > 1.8x average.
Short when price breaks below S1 AND price < 1d EMA34 AND volume > 1.8x average.
Exit when price returns to Camarilla pivot point (PP) or volume drops below average.
Camarilla pivots provide intraday support/resistance levels; EMA34 filters for higher timeframe trend.
Volume confirmation ensures breakout conviction. Designed for 12h timeframe targeting 50-150 total trades.
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
    
    # Load 1d data for EMA34 trend filter and Camarilla pivots - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivots from previous 1d bar
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        pp_val = pp_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d EMA34 AND volume spike
            if (price > r1_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1d EMA34 AND volume spike
            elif (price < s1_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot point OR volume drops below average
                if (price <= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot point OR volume drops below average
                if (price >= pp_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0