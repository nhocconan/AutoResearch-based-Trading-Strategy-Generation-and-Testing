#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with weekly trend filter and volume confirmation.
Long when Williams %R crosses above -80 from below (oversold reversal) AND price > weekly EMA34 (uptrend) AND volume > 1.5x average.
Short when Williams %R crosses below -20 from above (overbought reversal) AND price < weekly EMA34 (downtrend) AND volume > 1.5x average.
Exit on opposite Williams %R level (-20 for long, -80 for short) or trend reversal.
Williams %R identifies exhaustion points in both bull and bear markets, weekly EMA34 filters the major trend, volume confirms reversal strength.
Designed for 6h timeframe targeting 50-150 total trades over 4 years to avoid fee drag while capturing mean reversion moves.
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
    
    # Load weekly data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams %R (14-period) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1w_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price > weekly EMA34 AND volume spike
            if (i > 0 and williams_r[i-1] <= -80 and wr > -80 and 
                price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND price < weekly EMA34 AND volume spike
            elif (i > 0 and williams_r[i-1] >= -20 and wr < -20 and 
                  price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 OR trend reversal
                if (i > 0 and williams_r[i-1] >= -20 and wr < -20) or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -80 OR trend reversal
                if (i > 0 and williams_r[i-1] <= -80 and wr > -80) or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0