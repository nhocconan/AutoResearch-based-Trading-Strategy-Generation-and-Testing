#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND weekly close > weekly EMA34 AND volume > 2x average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND weekly close < weekly EMA34 AND volume > 2x average.
Exit when Alligator lines cross (jaws/teeth/lips lose alignment).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Williams Alligator catches trending moves; weekly trend filter avoids counter-trend trades in bear markets.
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams Alligator on primary timeframe (12h)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1w indicators to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close[i] > ema34_1w_aligned[i]  # using 12h close vs weekly EMA
        weekly_trend_down = close[i] < ema34_1w_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator conditions
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND weekly uptrend AND volume confirmation
            if bullish_alignment and weekly_trend_up and vol_current > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND weekly downtrend AND volume confirmation
            elif bearish_alignment and weekly_trend_down and vol_current > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross (loss of alignment)
            exit_signal = False
            
            if position == 1:
                # Exit long: Bullish alignment broken
                if not (jaw[i] < teeth[i] < lips[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bearish alignment broken
                if not (jaw[i] > teeth[i] > lips[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0