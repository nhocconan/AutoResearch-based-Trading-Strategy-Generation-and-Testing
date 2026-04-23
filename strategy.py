#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1w trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND weekly close > weekly EMA50 AND volume > 1.5x average.
Short when Bull Power < 0 AND Bear Power > 0 AND weekly close < weekly EMA50 AND volume > 1.5x average.
Exit when Elder Ray signals reverse OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Elder Ray measures bull/bear power relative to EMA13, working in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close_1w[-1] > ema50_1w[-1] if len(close_1w) == len(ema50_1w) else False  # placeholder, will use aligned
        weekly_trend_down = close_1w[-1] < ema50_1w[-1] if len(close_1w) == len(ema50_1w) else False
        # Actually use aligned values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        weekly_trend_up = weekly_close_aligned[i] > ema50_1w_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < ema50_1w_aligned[i]
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND weekly uptrend AND volume spike
            if (bull_val > 0 and bear_val < 0 and weekly_trend_up and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND weekly downtrend AND volume spike
            elif (bull_val < 0 and bear_val > 0 and weekly_trend_down and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Elder Ray reverses OR volume drops below average
                if (bull_val <= 0 or bear_val >= 0 or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Elder Ray reverses OR volume drops below average
                if (bull_val >= 0 or bear_val <= 0 or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0