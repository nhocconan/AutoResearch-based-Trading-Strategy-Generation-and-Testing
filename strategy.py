#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d (R1/S1) act as key support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and aligned 1d trend direction signal momentum continuation. 
Uses 12h timeframe with 1d Camarilla pivots, 1d EMA34 trend filter, and volume spike filter. 
Targets 50-150 total trades over 4 years (12-37/year) with controlled risk via position sizing.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (using previous day's OHLC) ---
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla calculations
    range_ = prev_day_high - prev_day_low
    camarilla_mult = 1.1 / 12
    r1 = prev_day_close + range_ * camarilla_mult * 1.1
    s1 = prev_day_close - range_ * camarilla_mult * 1.1
    r2 = prev_day_close + range_ * camarilla_mult * 1.5
    s2 = prev_day_close - range_ * camarilla_mult * 1.5
    
    # Align Camarilla levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # --- 1d EMA34 for trend filter ---
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for EMA and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Simple stoploss: 2.0x ATR from entry (using 12h bar range as proxy)
                atr_est = np.abs(high_12h[i] - low_12h[i])
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        if position == 0:
            # Look for entries: breakout with volume confirmation and trend alignment
            if vol_confirm:
                # Long: break above R1 with price above 1d EMA34 (uptrend)
                if close_12h[i] > r1_12h[i] and close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
                # Short: break below S1 with price below 1d EMA34 (downtrend)
                elif close_12h[i] < s1_12h[i] and close_12h[i] < ema34_12h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on break below S1 or trend reversal
                if close_12h[i] < s1_12h[i] or close_12h[i] < ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on break above R1 or trend reversal
                if close_12h[i] > r1_12h[i] or close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals