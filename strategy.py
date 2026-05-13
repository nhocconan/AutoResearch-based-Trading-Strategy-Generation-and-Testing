#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance levels.
Breakouts above R1 or below S1 with volume confirmation and 1-day EMA50 trend filter capture
trend moves in both bull and bear markets. Low trade frequency design targets 20-40 trades/year
to minimize fee drag while maintaining edge in trending markets.
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close to calculate today's levels
    phigh = np.concatenate([[high[0]], high[:-1]])  # Previous day high
    plow = np.concatenate([[low[0]], low[:-1]])    # Previous day low
    pclose = np.concatenate([[close[0]], close[:-1]])  # Previous day close
    
    # Camarilla calculations
    range_val = phigh - plow
    R1 = pclose + range_val * 1.1 / 12
    S1 = pclose - range_val * 1.1 / 12
    
    # Calculate 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend filter
            if close[i] > R1[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA50 (uptrend filter)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend filter
            elif close[i] < S1[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA50 (downtrend filter)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversal signal) or volume dries up
            if close[i] < S1[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversal signal) or volume dries up
            if close[i] > R1[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals