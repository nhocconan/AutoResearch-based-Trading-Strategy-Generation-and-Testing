#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot levels (R4/S4) with daily trend filter and volume confirmation.
In uptrend (price > daily EMA50), buy breakouts above weekly R4; in downtrend (price < daily EMA50), sell breakdowns below weekly S4.
Weekly pivots provide stronger support/resistance for longer-term trends. Volume > 1.5x 20-period average confirms breakout strength.
Exit on trend reversal or when price re-enters the weekly pivot range (R4 to S4).
Designed for 15-30 trades/year (60-120 total over 4 years) to minimize fee drag while capturing major trend moves.
Works in bull markets via R4 breakouts and in bear markets via S4 breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (Standard formula)
    # Pivot = (H + L + C) / 3
    # R4 = Close + 3*(High - Low)  (Aggressive extension)
    # S4 = Close - 3*(High - Low)  (Aggressive extension)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    weekly_r4 = close_1w + 3 * (high_1w - low_1w)
    weekly_s4 = close_1w - 3 * (high_1w - low_1w)
    
    # Align Weekly levels to 6h timeframe (wait for weekly bar to close)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Load daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above weekly R4 + uptrend + volume spike
            if (price_close > weekly_r4_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S4 + downtrend + volume spike
            elif (price_close < weekly_s4_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR price re-enters weekly pivot range (R4 to S4)
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # Re-entry exit: price moves back into weekly R4-S4 range
            if price_close <= weekly_r4_aligned[i] and price_close >= weekly_s4_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyR4S4_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0