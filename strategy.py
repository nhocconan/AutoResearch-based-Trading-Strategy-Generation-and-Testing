#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend with 1w HMA filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in ranging markets (2025 bearish bias).
# 1w HMA provides higher timeframe trend alignment to avoid counter-trend entries.
# Volume confirmation ensures breakout authenticity.
# Long when KAMA > prior KAMA AND price > 1w HMA AND volume > 1.5x 20-period average.
# Short when KAMA < prior KAMA AND price < 1w HMA AND volume > 1.5x 20-period average.
# Exit on ATR(14) trailing stop (2.0x) or contrary KAMA signal.
# Designed for BTC/ETH with strict entry to minimize trades and fee drag.

name = "1d_KAMA_1wHMA_VolumeSpike_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA(10,2,30) - ER=10, Fast=2, Slow=30
    close_s = pd.Series(close)
    change = np.abs(close - np.roll(close, 10))
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 30/30) + 30/30) ** 2  # (fast - slow) scaled
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w data for HMA21 trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA21 on 1w close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Volume filter: current 1d volume > 1.5x 20-period average (spike confirmation)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (i < 10 or np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if i > 0 and position == 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: KAMA rising AND price > 1w HMA AND volume spike
            if kama[i] > kama[i-1] and close[i] > hma_21_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: KAMA falling AND price < 1w HMA AND volume spike
            elif kama[i] < kama[i-1] and close[i] < hma_21_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR) OR KAMA turns down
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            kama_reversal = kama[i] < kama[i-1]
            if trailing_stop or kama_reversal:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR) OR KAMA turns up
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            kama_reversal = kama[i] > kama[i-1]
            if trailing_stop or kama_reversal:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals