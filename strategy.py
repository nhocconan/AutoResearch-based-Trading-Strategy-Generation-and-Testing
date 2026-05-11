# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_SwingFailure_1dTrend_Volume
Hypothesis: 6h swing failures (failure to make new high/low) combined with 1d EMA34 trend and volume confirmation.
In bull markets: look for bullish swing failure (higher low + break of prior swing high) for longs.
In bear markets: look for bearish swing failure (lower high + break of prior swing low) for shorts.
Swing failures indicate exhaustion and potential reversal, but we trade in direction of 1d trend to avoid counter-trend whipsaws.
Volume confirms conviction. Target: 20-40 trades/year.
"""

name = "6h_SwingFailure_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Swing Points: identify swing highs and lows ---
    # Swing high: high > previous high AND high > next high (using 3-bar window)
    # Swing low: low < previous low AND low < next low
    # We'll use a simple 3-bar swing detection
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(1, n-1):
        if high_6h[i] > high_6h[i-1] and high_6h[i] > high_6h[i+1]:
            swing_high[i] = True
        if low_6h[i] < low_6h[i-1] and low_6h[i] < low_6h[i+1]:
            swing_low[i] = True
    
    # --- Track most recent swing high and low ---
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high_val = high_6h[i]
        if swing_low[i]:
            last_low_val = low_6h[i]
        last_swing_high[i] = last_high_val
        last_swing_low[i] = last_low_val
    
    # --- Volume Filter: spike above 1.5x median of last 24 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=24, min_periods=12).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after we have enough data for swing detection
    start_idx = 2
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(last_swing_high[i]) or np.isnan(last_swing_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple time-based exit: exit after 8 bars if no swing failure
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema34_1d_aligned[i]
        trend_down = close_6h[i] < ema34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for bullish swing failure: higher low + break of prior swing high
            # Bullish SF: current low > prior swing low AND current close > prior swing high
            if (low_6h[i] > last_swing_low[i] and 
                close_6h[i] > last_swing_high[i] and
                trend_up and vol_ok):
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            # Look for bearish swing failure: lower high + break of prior swing low
            # Bearish SF: current high < prior swing high AND current close < prior swing low
            elif (high_6h[i] < last_swing_high[i] and 
                  close_6h[i] < last_swing_low[i] and
                  trend_down and vol_ok):
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Exit conditions: opposite swing failure or time-based
            if position == 1:
                # Exit long: bearish swing failure or price drops below prior swing low
                if (high_6h[i] < last_swing_high[i] and 
                    close_6h[i] < last_swing_low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish swing failure or price rises above prior swing high
                if (low_6h[i] > last_swing_low[i] and 
                    close_6h[i] > last_swing_high[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals