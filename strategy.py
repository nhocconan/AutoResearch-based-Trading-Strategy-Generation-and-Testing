#!/usr/bin/env python3
# 4h_orb_reversal_v1
# Hypothesis: Opening range breakout with reversal on 4h timeframe. Uses first 2 bars of 4h candle to establish range,
# then looks for reversal signals when price returns to range boundaries with volume confirmation.
# Designed to work in both bull and bear markets by capturing mean reversion within the daily range.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_orb_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for daily range and context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h opening range (first 2 bars of each 4h candle = 30 minutes)
    # We'll use the first 2 bars of each 4h candle to establish the opening range
    orb_high = np.full(n, np.nan)
    orb_low = np.full(n, np.nan)
    
    # For each 4h bar, the first 2 bars represent the opening range
    # Since we're on 4h timeframe, we need to look back to establish the range
    # We'll use the high/low of the current 4h bar as proxy for simplicity
    # In practice, we'd need intraday data, but we approximate with current bar
    
    # Daily range from 1d data
    daily_range = high_1d - low_1d
    avg_daily_range = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    
    # Current position within daily range
    # For each 4h bar, we need to know which 1d bar it belongs to
    # We'll use the 1d close as reference for trend
    
    # 1d EMA for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI on 4h for overbought/oversold
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or np.isnan(daily_range_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Calculate where current price sits in daily range
        # We approximate using the 1d bar that corresponds to this 4h period
        # For simplicity, we use the current daily range position
        
        # Since we don't have exact intraday, we use price action relative to recent swings
        # Look for swing points to establish range
        
        # Alternative approach: use recent high/low as range proxy
        lookback = 20
        if i < lookback:
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
            
        recent_high = np.max(high[i-lookback:i+1])
        recent_low = np.min(low[i-lookback:i+1])
        range_size = recent_high - recent_low
        
        if range_size == 0:
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
            
        # Position within range (0 = at low, 1 = at high)
        position_in_range = (close[i] - recent_low) / range_size
        
        # Volume filter
        volume_ok = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit when price reaches upper range or RSI overbought
            if position_in_range > 0.8 or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches lower range or RSI oversold
            if position_in_range < 0.2 or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Look for reversal at range boundaries with volume
            if position_in_range < 0.2 and rsi[i] < 30 and volume_ok:
                # Near low, oversold, volume confirmation -> long
                position = 1
                signals[i] = 0.25
            elif position_in_range > 0.8 and rsi[i] > 70 and volume_ok:
                # Near high, overbought, volume confirmation -> short
                position = -1
                signals[i] = -0.25
    
    return signals