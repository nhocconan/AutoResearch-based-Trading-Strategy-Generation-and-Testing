#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla pivot reversal with 1d EMA34 trend filter and volume confirmation.
In uptrend (price > 1d EMA34), buy reversals at S1 support; in downtrend (price < 1d EMA34), sell reversals at R1 resistance.
Volume must exceed 1.5x 20-period average to confirm reversal. Exit on trend reversal.
Designed for 15-35 trades/year (60-140 total over 4 years) to minimize fee drift while capturing mean-reversion in strong trends.
Works in bull markets via S1 bounces and in bear markets via R1 rejections with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivots (based on previous 4h bar)
    # Using typical price = (H+L+C)/3
    typical_price = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels: S1 = C - (H-L)*1.1/6, R1 = C + (H-L)*1.1/6
    camarilla_s1 = close_4h - (range_4h * 1.1 / 6)
    camarilla_r1 = close_4h + (range_4h * 1.1 / 6)
    
    # Align 4h Camarilla levels to 1h (wait for 4h bar to close)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        s1 = camarilla_s1_aligned[i]
        r1 = camarilla_r1_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price reverses up from S1 support + uptrend + volume spike
            if (price_close > s1 and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short: price reverses down from R1 resistance + downtrend + volume spike
            elif (price_close < r1 and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend reversal
            exit_signal = False
            
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_S1R1_1dEMA34_Volume_Session"
timeframe = "1h"
leverage = 1.0