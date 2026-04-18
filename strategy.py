#!/usr/bin/env python3
"""
4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend_ATRStop
Hypothesis: Camarilla pivot levels from daily timeframe act as key support/resistance levels.
Breakouts above R1 or below S1 with volume confirmation and 1-week EMA trend filter capture
institutional moves in both bull and bear markets. ATR-based stop loss limits drawdown.
Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price for previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formulas
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # Warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_1w_4h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1w_4h[i]
        
        if position == 0:
            # Long: break above R1 with volume in uptrend
            if price > R1_4h[i] and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 with volume in downtrend
            elif price < S1_4h[i] and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Maintain long until stop loss or reversal
            # ATR-based stop loss: exit if price drops 2*ATR below entry
            if price <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Reverse to short if price breaks below S1 with volume
            elif price < S1_4h[i] and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until stop loss or reversal
            # ATR-based stop loss: exit if price rises 2*ATR above entry
            if price >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Reverse to long if price breaks above R1 with volume
            elif price > R1_4h[i] and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend_ATRStop"
timeframe = "4h"
leverage = 1.0