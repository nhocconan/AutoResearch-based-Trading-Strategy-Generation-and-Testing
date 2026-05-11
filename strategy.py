#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from the daily timeframe act as strong support/resistance. 
A break above R1 or below S1 with volume confirmation and alignment with the 1-day EMA trend 
signals momentum continuation. Uses 4h timeframe with 1d Camarilla levels and 1d EMA50 trend filter.
Target: 20-50 trades per year to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Previous day's OHLC for Camarilla calculation ---
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    # Handle first value
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    
    # Camarilla levels calculation
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    # We use R1 and S1 as key levels
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + (camarilla_range * 1.0833)
    s1 = prev_day_close - (camarilla_range * 1.0833)
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d EMA50 for trend filter ---
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry (using 4h range as proxy)
                atr_est = np.abs(high_4h[i] - low_4h[i])
                if position == 1 and close_4h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for entries: breakout with trend and volume
            if vol_confirm:
                # Long: break above R1 with price above EMA50 (uptrend)
                if close_4h[i] > r1_4h[i] and close_4h[i] > ema50_4h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                # Short: break below S1 with price below EMA50 (downtrend)
                elif close_4h[i] < s1_4h[i] and close_4h[i] < ema50_4h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long: exit on break below S1 or trend reversal
                if close_4h[i] < s1_4h[i] or close_4h[i] < ema50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit on break above R1 or trend reversal
                if close_4h[i] > r1_4h[i] or close_4h[i] > ema50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals