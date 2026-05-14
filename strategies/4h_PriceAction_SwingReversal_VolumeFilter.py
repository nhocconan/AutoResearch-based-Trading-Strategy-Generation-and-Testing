#!/usr/bin/env python3
"""
4h_PriceAction_SwingReversal_VolumeFilter
Hypothesis: Capture swing reversals at key price levels using price action (close above/below prior swing high/low) combined with volume confirmation and 12h trend filter. Works in both bull and bear markets by trading mean reversion within the trend context. Uses swing points for dynamic support/resistance, reducing false breakouts. Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_PriceAction_SwingReversal_VolumeFilter"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === Swing Points (5-bar lookback) ===
    # Swing high: high[i] is highest in [i-2, i-1, i, i+1, i+2]
    # Swing low: low[i] is lowest in [i-2, i-1, i, i+1, i+2]
    window = 5
    half = window // 2
    
    # Calculate swing highs and lows using rolling window
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Rolling max/min with center alignment
    rolling_max = high_series.rolling(window=window, center=True, min_periods=window).max()
    rolling_min = low_series.rolling(window=window, center=True, min_periods=window).min()
    
    # Swing points: where price equals the rolling extreme
    is_swing_high = (high == rolling_max.values)
    is_swing_low = (low == rolling_min.values)
    
    # Extract swing levels
    swing_high = np.where(is_swing_high, high, np.nan)
    swing_low = np.where(is_swing_low, low, np.nan)
    
    # Forward fill to get last swing level
    swing_high = pd.Series(swing_high).ffill().values
    swing_low = pd.Series(swing_low).ffill().values
    
    # === 12h EMA25 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    ema25_12h_4h = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers swing calculation and EMA)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or 
            np.isnan(ema25_12h_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above prior swing low (support bounce) + above 12h EMA + volume spike
            if (close[i] > swing_low[i] and 
                close[i] > ema25_12h_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Close crosses below prior swing high (resistance rejection) + below 12h EMA + volume spike
            elif (close[i] < swing_high[i] and 
                  close[i] < ema25_12h_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (8 bars)
            holding_bars += 1
            if holding_bars < 8:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Close crosses back through the swing level in opposite direction
            if position == 1:
                if close[i] < swing_low[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > swing_high[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals