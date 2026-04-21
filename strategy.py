#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation.
Longs when %R < -80 (oversold) with 12h EMA50 uptrend and volume > 1.3x average;
Shorts when %R > -20 (overbought) with 12h EMA50 downtrend and volume > 1.3x average.
Exit when %R crosses back through -50 (mean) or 2x ATR stop.
Williams %R identifies extremes in 6-bar cycles, effective in ranging markets.
Designed for 15-30 trades/year to minimize fee decay while capturing reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA for 12h trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R calculation (14-period) on 6m data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * (highest_high - close_6h) / rr
    
    # Volume confirmation: volume spike > 1.3x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: oversold with uptrend and volume
            if (williams_r_val < -80 and 
                ema_trend > 0 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought with downtrend and volume
            elif (williams_r_val > -20 and 
                  ema_trend < 0 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: mean reversion (-50 cross) OR ATR-based stoploss
            exit_signal = False
            
            # Mean reversion exit: %R crosses back through -50
            if position == 1 and williams_r_val > -50:
                exit_signal = True
            elif position == -1 and williams_r_val < -50:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry area)
            if position == 1:
                # For longs, stop below recent low
                if prices['low'].iloc[i] < prices['low'].rolling(5).min().iloc[i] - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above recent high
                if prices['high'].iloc[i] > prices['high'].rolling(5).max().iloc[i] + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_12hEMA50_Trend_Volume1.3x_ATR2x"
timeframe = "6h"
leverage = 1.0