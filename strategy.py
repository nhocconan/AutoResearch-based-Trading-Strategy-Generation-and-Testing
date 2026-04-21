#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 12h trend filter and volume confirmation.
Longs when Williams %R < -80 (oversold) and 12h EMA50 trending up with volume > 1.3x average.
Shorts when Williams %R > -20 (overbought) and 12h EMA50 trending down with volume > 1.3x average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies overextended moves likely to reverse; 12h EMA50 filters for trend alignment.
Designed for 20-35 trades/year to minimize fee drift while capturing high-probability reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 50-period EMA on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 4h data
    highest_high = pd.Series(prices['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: oversold + bullish 12h trend + volume
            if (williams_r_val < -80 and 
                price_close > ema_50_12h_val and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + bearish 12h trend + volume
            elif (williams_r_val > -20 and 
                  price_close < ema_50_12h_val and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R crosses back above -50 (long) or below -50 (short)
            exit_signal = False
            
            if position == 1 and williams_r_val > -50:
                exit_signal = True
            elif position == -1 and williams_r_val < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_12hEMA50_Trend_Volume1.3x"
timeframe = "4h"
leverage = 1.0