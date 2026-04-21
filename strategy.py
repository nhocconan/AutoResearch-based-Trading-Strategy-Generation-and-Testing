#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day Camarilla pivot levels (R1/S1) with 1-day EMA34 trend filter and volume spike confirmation.
In uptrend (price > 1d EMA34), buy breakouts above 1d Camarilla R1; in downtrend (price < 1d EMA34), sell breakdowns below 1d Camarilla S1.
Volume must exceed 2.5x 20-period average to confirm breakout strength. Exit on trend reversal or 2.0x ATR stop.
Designed for 15-30 trades/year (60-120 total over 4 years) to minimize fee drag while capturing major trend moves.
Works in bull markets via R1 breakouts and in bear markets via S1 breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation (volume spike > 2.5x 20-period average)
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above 1d Camarilla R1 + uptrend + volume spike
            if (price_close > r1_val and 
                price_close > ema_trend and 
                vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1d Camarilla S1 + downtrend + volume spike
            elif (price_close < s1_val and 
                  price_close < ema_trend and 
                  vol_ratio_val > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (2.0x ATR from entry)
            if position == 1:
                # Approximate entry price as the R1 breakout level
                entry_approx = r1_aligned[i-1] if i > 0 else r1_aligned[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry price as the S1 breakdown level
                entry_approx = s1_aligned[i-1] if i > 0 else s1_aligned[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0