#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 Reversal pattern with 1d trend filter and volume confirmation.
# 123 Reversal: Price makes higher high/low, then pullback, then breaks prior extreme.
# Long: Higher low forms, pullback to prior swing low, breaks above swing high with volume.
# Short: Higher high forms, pullback to prior swing high, breaks below swing low with volume.
# 1d EMA filter ensures alignment with higher timeframe trend.
# Target: 20-50 trades/year for 4h timeframe to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter (longer term for stability)
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Swing points detection (lookback 5 bars)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    lookback = 5
    
    for i in range(lookback, n):
        # Swing high: highest high in lookback window
        swing_high[i] = np.max(high[i-lookback:i])
        # Swing low: lowest low in lookback window
        swing_low[i] = np.min(low[i-lookback:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long setup: Higher low forms, then breaks above swing high with volume
            # Condition: Current low > prior swing low AND breaks above swing high
            if (low[i] > swing_low[i] and 
                price > swing_high[i] and
                price > ema_trend and  # Above 1d EMA for uptrend bias
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short setup: Higher high forms, then breaks below swing low with volume
            # Condition: Current high < prior swing high AND breaks below swing low
            elif (high[i] < swing_high[i] and 
                  price < swing_low[i] and
                  price < ema_trend and  # Below 1d EMA for downtrend bias
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below swing low or trend changes
            if (price < swing_low[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above swing high or trend changes
            if (price > swing_high[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_123Reversal_Trend_Volume"
timeframe = "4h"
leverage = 1.0