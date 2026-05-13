#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide strong intraday support/resistance.
Breakouts above R3 or below S3 with volume confirmation and 1-day trend filter capture
trends in both bull and bear markets. Designed for low trade frequency (12-37/year) with
clear entry/exit rules to minimize fee drag.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day (HLC of previous day)
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    camarilla_high = close + ((high - low) * 1.25)
    camarilla_low = close - ((high - low) * 1.25)
    
    # Get 1-day trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend filter
            if close[i] > camarilla_high[i] and volume_confirm[i]:
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend filter
            elif close[i] < camarilla_low[i] and volume_confirm[i]:
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below previous day's close or RSI-like condition (price < VWAP approximation)
            # Use simple exit: price breaks below the pivot point (close of previous day approximated)
            pivot_point = (high[i-1] + low[i-1] + close[i-1]) / 3.0  # approximate from previous bar
            if close[i] < pivot_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above previous day's close
            pivot_point = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            if close[i] > pivot_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals