#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based breakout with volume confirmation and 12h EMA trend filter
# Uses volatility-adjusted breakouts to capture strong moves in both bull and bear markets
# Volume filter ensures breakouts have participation, 12h EMA aligns with intermediate trend
# Target: 20-50 trades/year to minimize fee drag while maintaining edge
name = "4h_ATRBreakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for multi-timeframe analysis (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 4h ATR for volatility-based breakout levels
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4-period ATR multiplier for breakout threshold
    atr_mult = 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        if position == 0:
            # Long: breakout above previous close + ATR multiple + volume + 12h uptrend
            if high[i] > close[i-1] + atr_mult * atr_val and volume_filter and price > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below previous close - ATR multiple + volume + 12h downtrend
            elif low[i] < close[i-1] - atr_mult * atr_val and volume_filter and price < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below previous close - ATR multiple or trailing stop
            if close[i] < close[i-1] - atr_mult * atr_val or close[i] < highest_since_entry - 1.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update highest close since entry for trailing stop
                if 'highest_since_entry' not in locals():
                    highest_since_entry = close[i]
                else:
                    highest_since_entry = max(highest_since_entry, close[i])
        
        elif position == -1:
            # Exit: price closes above previous close + ATR multiple or trailing stop
            if close[i] > close[i-1] + atr_mult * atr_val or close[i] > lowest_since_entry + 1.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Update lowest close since entry for trailing stop
                if 'lowest_since_entry' not in locals():
                    lowest_since_entry = close[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, close[i])
    
    return signals