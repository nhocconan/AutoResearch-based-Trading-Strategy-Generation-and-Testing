#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly RSI divergence and volume confirmation.
# Uses weekly RSI(14) divergences for mean-reversion entries and volume spikes for confirmation.
# Weekly RSI identifies overbought/oversold conditions on higher timeframe, while volume
# confirms momentum exhaustion. Works in both bull and bear markets by fading extremes.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_WeeklyRSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Identify swing highs/lows for divergence detection (using weekly data)
    # Swing high: higher high followed by lower high
    # Swing low: lower low followed by higher low
    swing_high = np.zeros_like(rsi, dtype=bool)
    swing_low = np.zeros_like(rsi, dtype=bool)
    
    for i in range(2, len(rsi) - 2):
        # Swing high: current high > previous high and current high > next high
        if (rsi[i] > rsi[i-1] and rsi[i] > rsi[i-2] and 
            rsi[i] > rsi[i+1] and rsi[i] > rsi[i+2]):
            swing_high[i] = True
        # Swing low: current low < previous low and current low < next low
        if (rsi[i] < rsi[i-1] and rsi[i] < rsi[i-2] and 
            rsi[i] < rsi[i+1] and rsi[i] < rsi[i+2]):
            swing_low[i] = True
    
    # Align swing points to 6t timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1w, swing_high.astype(float))
    swing_low_aligned = align_htf_to_ltf(prices, df_1w, swing_low.astype(float))
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bearish divergence: price makes higher high, RSI makes lower high
            # Look for swing highs in price and RSI
            bearish_div = False
            bullish_div = False
            
            # Check for bearish divergence (for short entries)
            if swing_high_aligned[i] > 0.5:  # Current bar is a swing high
                # Look back for previous swing high
                for j in range(i-1, max(0, i-50), -1):  # Look back up to 50 bars
                    if swing_high_aligned[j] > 0.5:
                        # Found previous swing high, check for divergence
                        if (high[i] > high[j] and  # Price made higher high
                            rsi_aligned[i] < rsi_aligned[j]):  # RSI made lower high
                            bearish_div = True
                        break
            
            # Check for bullish divergence (for long entries)
            if swing_low_aligned[i] > 0.5:  # Current bar is a swing low
                # Look back for previous swing low
                for j in range(i-1, max(0, i-50), -1):  # Look back up to 50 bars
                    if swing_low_aligned[j] > 0.5:
                        # Found previous swing low, check for divergence
                        if (low[i] < low[j] and  # Price made lower low
                            rsi_aligned[i] > rsi_aligned[j]):  # RSI made higher low
                            bullish_div = True
                        break
            
            # Enter short on bearish divergence with volume confirmation
            if bearish_div and volume_filter[i] and rsi_aligned[i] > 50:
                signals[i] = -0.25
                position = -1
            # Enter long on bullish divergence with volume confirmation
            elif bullish_div and volume_filter[i] and rsi_aligned[i] < 50:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or bullish divergence fails
            if rsi_aligned[i] >= 50 or not bullish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or bearish divergence fails
            if rsi_aligned[i] <= 50 or not bearish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals