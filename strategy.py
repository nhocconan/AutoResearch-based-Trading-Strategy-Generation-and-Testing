#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla pivot R1/S1 levels breakout with volume confirmation and 4h trend filter.
Long when price breaks above 4h Camarilla R1 level AND volume > 1.3x 20-period average AND 4h EMA20 > EMA50.
Short when price breaks below 4h Camarilla S1 level AND volume > 1.3x 20-period average AND 4h EMA20 < EMA50.
Exit when price retraces to the 4h Camarilla midpoint (pivot point) or opposite Camarilla level is touched.
Uses discrete position sizing (0.20) to control drawdown and fee churn.
Designed for 1h timeframe to target 15-37 trades/year per symbol (60-150 total over 4 years).
Uses 4h for signal direction and regime, 1h only for entry timing precision.
Includes session filter (08-20 UTC) to avoid low-liquidity periods.
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
    
    # Calculate 4h Camarilla pivot levels (R1, S1, midpoint/pivot point)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Camarilla levels based on previous 4h bar's OHLC
    # R1 = Close + 1.1*(High - Low)/1.25
    # S1 = Close - 1.1*(High - Low)/1.25
    # Pivot point = (High + Low + Close)/3
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 1.25
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 1.25
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # 4h EMA20 and EMA50 for trend filter
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 2)  # EMA50 needs 50, volume MA needs 20, 4h data needs at least 2 for shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        ema20_val = ema20_4h_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        hour = hours[i]
        
        # Session filter: only trade during 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: Price breaks above 4h Camarilla R1 AND volume spike AND 4h uptrend
            if (price > r1_val and volume[i] > 1.3 * vol_ma_val and ema20_val > ema50_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Camarilla S1 AND volume spike AND 4h downtrend
            elif (price < s1_val and volume[i] > 1.3 * vol_ma_val and ema20_val < ema50_val):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 4h Camarilla pivot point
            if position == 1 and price <= pivot_val:
                exit_signal = True
            elif position == -1 and price >= pivot_val:
                exit_signal = True
            
            # Secondary exit: Opposite Camarilla level touched (strong reversal signal)
            if position == 1 and price < s1_val:
                exit_signal = True
            elif position == -1 and price > r1_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_VolumeConfirmation_4hEMAFilter_Session"
timeframe = "1h"
leverage = 1.0