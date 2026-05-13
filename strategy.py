#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray + 1d trend filter. 
# Long when: price > Alligator Jaw, Bull Power > 0, and price > 1d EMA50.
# Short when: price < Alligator Jaw, Bear Power < 0, and price < 1d EMA50.
# Exit on close crossing Alligator Teeth (reversal signal).
# Uses Williams Alligator (Jaw/Teeth/Lips) from Elder's system to identify trend,
# Elder Ray to measure bull/bear power behind the move, and 1d EMA50 for higher-timeframe trend alignment.
# Designed for 6h timeframe to capture intermediate trends with controlled trade frequency.

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator: SMAs of median price (typical price) with different periods
    # Jaw: 13-period SMMA, Teeth: 8-period, Lips: 5-period
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    typical_price = (high + low + close) / 3.0
    
    # Jaw (13-period SMMA)
    jaw = pd.Series(typical_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    # Teeth (8-period SMMA)
    teeth = pd.Series(typical_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    # Lips (5-period SMMA)
    lips = pd.Series(typical_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA50 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Jaw, Bull Power > 0, and price > 1d EMA50
            if close[i] > jaw[i] and bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Jaw, Bear Power < 0, and price < 1d EMA50
            elif close[i] < jaw[i] and bear_power[i] < 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Teeth (trend weakening)
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Teeth (trend weakening)
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals