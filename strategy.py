#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with daily volume spike and weekly trend filter.
# Uses daily Camarilla levels (H4/L4) for mean reversion entries. Long when price crosses above L4
# with volume spike and weekly uptrend. Short when price crosses below H4 with volume spike and weekly downtrend.
# Designed to capture reversals in ranging markets while avoiding counter-trend trades in strong trends.
name = "12h_Camarilla_H4L4_Reversal_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Weekly EMA trend filter
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: volume > 1.5x 24-period EMA (24*12h = 12 days)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_confirm = volume > (1.5 * vol_ema24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ema24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price crosses above L4 with volume spike and weekly uptrend
            if (price > camarilla_l4_aligned[i] and 
                close[i-1] <= camarilla_l4_aligned[i-1] and
                vol_confirm[i] and price > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below H4 with volume spike and weekly downtrend
            elif (price < camarilla_h4_aligned[i] and 
                  close[i-1] >= camarilla_h4_aligned[i-1] and
                  vol_confirm[i] and price < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below L4 or weekly trend turns down
            if price < camarilla_l4_aligned[i] or price < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above H4 or weekly trend turns up
            if price > camarilla_h4_aligned[i] or price > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals