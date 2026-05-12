#!/usr/bin/env python3
name = "1d_Wilson_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Wilson Bands: center = EMA(14), width = ATR(7) * 1.5
    close_s = pd.Series(close)
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(14)
    wilson_mid = close_s.ewm(span=14, adjust=False, min_periods=14).mean().values
    # ATR(7)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr7 = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    width = atr7 * 1.5
    wilson_upper = wilson_mid + width
    wilson_lower = wilson_mid - width
    
    # Weekly trend filter: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Volume confirmation: volume > 1.2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(wilson_upper[i]) or np.isnan(wilson_lower[i]) or np.isnan(weekly_ema21_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above upper band + weekly trend up + volume
            if (close[i] > wilson_upper[i]) and (close[i] > weekly_ema21_aligned[i]) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below lower band + weekly trend down + volume
            elif (close[i] < wilson_lower[i]) and (close[i] < weekly_ema21_aligned[i]) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below middle band or weekly trend down
            if (close[i] < wilson_mid[i]) or (close[i] < weekly_ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above middle band or weekly trend up
            if (close[i] > wilson_mid[i]) or (close[i] > weekly_ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals