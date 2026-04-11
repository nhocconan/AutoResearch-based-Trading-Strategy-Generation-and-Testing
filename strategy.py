#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and Bear Power < 0 with volume > 1.5x average and 1d EMA(50) bullish.
# Short when Bear Power < 0 and Bull Power < 0 with volume > 1.5x average and 1d EMA(50) bearish.
# Uses weekly pivot levels to filter entries: only trade when price is above weekly pivot (long) or below (short).
# Designed for 12-37 trades/year on 6h timeframe with focus on institutional participation.
# Volume filter reduces false signals, weekly pivot adds structural bias.

name = "6h_1d_1w_elder_ray_volume_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We'll use the pivot as the main reference level
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1d trend direction
        is_bullish_trend = close[i] > ema_50_1d_aligned[i]
        is_bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Determine price relative to weekly pivot
        above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0
        
        # Entry conditions
        bullish_entry = (strong_bull_power and strong_bear_power and  # Both conditions indicate strength
                        vol_filter and is_bullish_trend and above_weekly_pivot)
        
        bearish_entry = (strong_bear_power and 
                        vol_filter and is_bearish_trend and below_weekly_pivot)
        
        # Exit conditions: opposite signal or loss of momentum
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bear power becomes positive (loss of selling pressure) or trend turns bearish
            exit_long = (bear_power[i] >= 0) or (not is_bullish_trend)
        elif position == -1:
            # Exit short if bull power becomes negative (loss of buying pressure) or trend turns bullish
            exit_short = (bull_power[i] <= 0) or (not is_bearish_trend)
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals