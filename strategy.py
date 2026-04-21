#!/usr/bin/env python3
"""
4h_Momentum_Oscillator_1dTrend_Confirmation
Hypothesis: Use 4h momentum oscillator (Williams %R) to identify overbought/oversold conditions, confirmed by 1d EMA50 trend direction. Enter when momentum reverses from extreme levels in direction of higher timeframe trend. Exit on opposite extreme or trend change. Designed to capture mean-reversion within trend, working in both bull and bear markets by following 1d trend while using 4h momentum for timing. Target 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams %R on 4h (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold + price above 1d EMA50
            if (wr > -80 and 
                williams_r[i-1] <= -80 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought + price below 1d EMA50
            elif (wr < -20 and 
                  williams_r[i-1] >= -20 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Williams %R reaches opposite extreme
            if position == 1 and wr >= -20:
                signals[i] = 0.0
                position = 0
            elif position == -1 and wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Momentum_Oscillator_1dTrend_Confirmation"
timeframe = "4h"
leverage = 1.0