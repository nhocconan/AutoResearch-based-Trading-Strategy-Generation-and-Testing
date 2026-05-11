#!/usr/bin/env python3
# 4h_RCI_Reversal_1dTrend
# Hypothesis: Combines 4h Relative Strength Index (RSI) with 1d trend structure and volume confirmation.
# Long when: 1) daily structure is bullish (HH and HL), 2) RSI(14) crosses above 30 from below, 3) volume > 1.5x 20-period average.
# Short when: 1) daily structure is bearish (LH and LL), 2) RSI(14) crosses below 70 from above, 3) volume > 1.5x 20-period average.
# Exits when RSI returns to 50 or structure breaks.
# Works in bull markets by buying pullbacks in uptrends and in bear markets by selling rallies in downtrends.
# RSI provides mean-reversion signals within the trend, reducing false breakouts. 4h timeframe limits trades to avoid fee drag.

name = "4h_RCI_Reversal_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- 4h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align all 1d indicators to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI(14) and volume MA(20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Structure from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        # RSI conditions
        rsi_above_30 = rsi[i] > 30
        rsi_below_70 = rsi[i] < 70
        rsi_crossed_up_30 = (rsi[i] > 30) & (rsi[i-1] <= 30) if i > 0 else False
        rsi_crossed_down_70 = (rsi[i] < 70) & (rsi[i-1] >= 70) if i > 0 else False
        
        if position == 0:
            if is_uptrend and vol_spike and rsi_crossed_up_30:
                # Long: daily uptrend + volume spike + RSI crosses above 30
                signals[i] = 0.25
                position = 1
            elif is_downtrend and vol_spike and rsi_crossed_down_70:
                # Short: daily downtrend + volume spike + RSI crosses below 70
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RSI returns to 50 OR structure breaks down
                if rsi[i] >= 50 or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to 50 OR structure breaks up
                if rsi[i] <= 50 or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals