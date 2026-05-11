#!/usr/bin/env python3
# 4h_Keltner_Breakout_1dTrend_Volume
# Hypothesis: Combines 4h Keltner channel breakouts with 1d trend structure and volume confirmation.
# Long when: 1) daily structure is bullish (HH and HL), 2) price breaks above 4h Keltner upper band (EMA20 + 2*ATR), 3) volume > 1.5x 20-period average.
# Short when: 1) daily structure is bearish (LH and LL), 2) price breaks below 4h Keltner lower band (EMA20 - 2*ATR), 3) volume > 1.5x 20-period average.
# Exits when price returns to the EMA20 or structure breaks.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
# Volume confirmation reduces false breakouts. 4h timeframe limits trades to avoid fee drag.

name = "4h_Keltner_Breakout_1dTrend_Volume"
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
    
    # --- 4h EMA20 for Keltner center ---
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 4h ATR(20) for Keltner width ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first bar has no previous close
    atr_series = pd.Series(tr)
    atr20 = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 4h Keltner bands ---
    keltner_upper = ema20 + 2 * atr20
    keltner_lower = ema20 - 2 * atr20
    
    # --- 4h volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align all 1d indicators to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA20, ATR20, and volume MA(20)
    start_idx = max(20, 20)  # EMA20, ATR20, vol MA all need 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or
            np.isnan(ema20[i]) or
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
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price above Keltner upper
                if close[i] > keltner_upper[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price below Keltner lower
                if close[i] < keltner_lower[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to EMA20 OR structure breaks down
                if close[i] < ema20[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA20 OR structure breaks up
                if close[i] > ema20[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals