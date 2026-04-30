#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Uses 6h timeframe to capture medium-term breakouts with lower frequency than 4h
# Williams Fractals identify potential reversal/breakout points (requires 2-bar confirmation)
# 1d EMA34 provides trend filter to avoid counter-trend trades in bear markets
# Volume confirmation >1.5x 20-period average reduces false breakouts
# Discrete position sizing: 0.25 for entries to limit fee drag
# Works in both bull and bear markets: trend filter aligns with higher timeframe direction

name = "6h_WilliamsFractal_Breakout_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams Fractals (requires 5-bar window: 2 left, center, 2 right)
    # Bullish fractal: low[l] < low[l-1] and low[l] < low[l-2] and low[l] < low[l+1] and low[l] < low[l+2]
    # Bearish fractal: high[h] > high[h-1] and high[h] > high[h-2] and high[h] > high[h+1] and high[h] > high[h+2]
    # We'll use the high/low of the fractal point as breakout levels
    
    bullish_fractal = np.full(n, np.nan)
    bearish_fractal = np.full(n, np.nan)
    
    # Calculate fractals (avoiding look-ahead by using only past data)
    for i in range(2, n-2):
        # Bullish fractal at i: low[i] is lowest of i-2,i-1,i,i+1,i+2
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]
        # Bearish fractal at i: high[i] is highest of i-2,i-1,i,i+1,i+2
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Williams fractal needs 2 extra 1d bars for confirmation (center bar + 2 more to confirm)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup for volume MA (20), EMA (34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Check for bullish fractal breakout (price breaks above recent bullish fractal low)
        bullish_breakout = False
        if i >= 2 and not np.isnan(bullish_fractal[i-2]):
            # Price breaks above the bullish fractal low from 2 bars ago (to allow for confirmation)
            bullish_breakout = curr_close > bullish_fractal[i-2]
        
        # Check for bearish fractal breakout (price breaks below recent bearish fractal high)
        bearish_breakout = False
        if i >= 2 and not np.isnan(bearish_fractal[i-2]):
            # Price breaks below the bearish fractal high from 2 bars ago
            bearish_breakout = curr_close < bearish_fractal[i-2]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above bullish fractal + above 1d EMA34
                if bullish_breakout and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below bearish fractal + below 1d EMA34
                elif bearish_breakout and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below the most recent bullish fractal (stop loss) 
            # or above the most recent bearish fractal (take profit at opposite fractal)
            exit_long = False
            if i >= 2 and not np.isnan(bullish_fractal[i-2]):
                # Stop loss: price breaks below bullish fractal low
                if curr_close < bullish_fractal[i-2]:
                    exit_long = True
            if i >= 2 and not np.isnan(bearish_fractal[i-2]):
                # Take profit: price reaches bearish fractal level (opposite fractal)
                if curr_close > bearish_fractal[i-2]:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above the most recent bearish fractal (stop loss)
            # or below the most recent bullish fractal (take profit at opposite fractal)
            exit_short = False
            if i >= 2 and not np.isnan(bearish_fractal[i-2]):
                # Stop loss: price breaks above bearish fractal high
                if curr_close > bearish_fractal[i-2]:
                    exit_short = True
            if i >= 2 and not np.isnan(bullish_fractal[i-2]):
                # Take profit: price reaches bullish fractal level (opposite fractal)
                if curr_close < bullish_fractal[i-2]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals