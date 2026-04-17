#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 reversal pattern with volume confirmation and 1d trend filter.
# The 123 reversal is a price action pattern where:
# 1. Point 1: swing high/low
# 2. Point 2: retracement
# 3. Point 3: failure to exceed point 1, then break in opposite direction
# Long: Higher low (point 2 > point 1 low) then break above point 1 high
# Short: Lower high (point 2 < point 1 high) then break below point 1 low
# Uses 1d EMA50 for trend filter and volume spike for confirmation.
# Designed to work in both trending and ranging markets with low turnover.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h swing points for 123 pattern
    # Look for swing highs and lows over 5-period window
    window = 5
    high_roll_max = pd.Series(high_4h).rolling(window=window, center=True).max().values
    low_roll_min = pd.Series(low_4h).rolling(window=window, center=True).min().values
    
    # Identify swing points (where price equals rolling max/min)
    is_swing_high = (high_4h == high_roll_max)
    is_swing_low = (low_4h == low_roll_min)
    
    # Arrays to store recent swing points
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    # Track most recent swing points
    for i in range(n):
        if is_swing_high[i]:
            last_swing_high[i] = high_4h[i]
        elif i > 0:
            last_swing_high[i] = last_swing_high[i-1]
            
        if is_swing_low[i]:
            last_swing_low[i] = low_4h[i]
        elif i > 0:
            last_swing_low[i] = last_swing_low[i-1]
    
    # Align swing points to 4h
    last_swing_high_4h = align_htf_to_ltf(prices, df_4h, last_swing_high)
    last_swing_low_4h = align_htf_to_ltf(prices, df_4h, last_swing_low)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need enough data for swing detection and filters
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(last_swing_high_4h[i]) or 
            np.isnan(last_swing_low_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        # Get recent swing points (look back 1-5 bars to avoid look-ahead)
        lookback = 5
        start_lookback = max(0, i - lookback)
        
        # Find most recent swing high and low before current bar
        recent_highs = last_swing_high_4h[start_lookback:i]
        recent_lows = last_swing_low_4h[start_lookback:i]
        
        # Filter out NaN values
        valid_highs = recent_highs[~np.isnan(recent_highs)]
        valid_lows = recent_lows[~np.isnan(recent_lows)]
        
        if len(valid_highs) == 0 or len(valid_lows) == 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        last_high = valid_highs[-1]
        last_low = valid_lows[-1]
        
        # 123 Long pattern:
        # 1. Point 1: swing low (last_low)
        # 2. Point 2: retracement higher than point 1 (we look for higher low in recent price action)
        # 3. Point 3: failure to make new low, then break above point 1
        # Simplified: current price above last swing low AND showing higher low characteristic
        higher_low_condition = low[i] > last_low  # Higher low than point 1
        break_above = close[i] > last_high  # Break above point 1 (swing high)
        
        # 123 Short pattern:
        # 1. Point 1: swing high (last_high)
        # 2. Point 2: retracement lower than point 1 (lower high)
        # 3. Point 3: failure to make new high, then break below point 1
        lower_high_condition = high[i] < last_high  # Lower high than point 1
        break_below = close[i] < last_low  # Break below point 1 (swing low)
        
        if position == 0:
            # Long: Higher low break above swing high with volume and above EMA
            if (higher_low_condition and break_above and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: Lower high break below swing low with volume and below EMA
            elif (lower_high_condition and break_below and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below swing low OR below EMA
            if (close[i] < last_low) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above swing high OR above EMA
            if (close[i] > last_high) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_123_Reversal_Volume_EMA50"
timeframe = "4h"
leverage = 1.0