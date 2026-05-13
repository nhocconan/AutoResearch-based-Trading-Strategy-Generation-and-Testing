#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3, S3) on 12h timeframe breakouts with 1-day trend filter and volume confirmation capture institutional breakout moves in both bull and bear markets.
Designed for low trade frequency (15-25/year) with trend-following logic that adapts to market regimes using higher timeframe filters.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Calculate 12h Camarilla pivot levels using previous 12h bar's high, low, close
    # For first bar, use current bar's values (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    R3 = prev_close + range_val * 1.1 / 2
    S3 = prev_close - range_val * 1.1 / 2
    
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
            if close[i] > R3[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA34 (uptrend filter)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend filter
            elif close[i] < S3[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA34 (downtrend filter)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversion to mean) or volume drops
            if close[i] < S3[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversion to mean) or volume drops
            if close[i] > R3[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals