#!/usr/bin/env python3
# 6h_12h_1d_williams_fractal_volume_v1
# Strategy: 6s Williams Fractal breakout with volume confirmation and 12h/1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Williams Fractals identify key support/resistance levels. Breakouts above/below
# fractals with volume confirmation and aligned 12h/1d trend capture sustained moves.
# Works in bull/bear by following trend direction via 12h EMA50 and 1d EMA200 regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_williams_fractal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d EMA(200) for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams Fractals (5-bar window: 2 left, 2 right)
    # Bearish fractal: high[n-2] is highest of [n-4, n-3, n-2, n-1, n]
    # Bullish fractal: low[n-2] is lowest of [n-4, n-3, n-2, n-1, n]
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = True
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = True
    
    # Value arrays: fractal levels (0 where no fractal)
    bearish_level = np.where(bearish_fractal, high, 0.0)
    bullish_level = np.where(bullish_fractal, low, 0.0)
    
    # Forward fill to get most recent fractal level
    bearish_level_ff = np.where(bearish_level != 0, bearish_level, np.nan)
    bullish_level_ff = np.where(bullish_level != 0, bullish_level, np.nan)
    
    # Create pandas series for ffill, then convert back
    bearish_ff = pd.Series(bearish_level_ff).ffill().fillna(0).values
    bullish_ff = pd.Series(bullish_level_ff).ffill().fillna(0).values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(5, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(bearish_ff[i]) or np.isnan(bullish_ff[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get most recent fractal levels (avoid look-ahead by using past values)
        # We use i-1 to ensure we only use completed bars
        recent_bearish = bearish_ff[i-1]
        recent_bullish = bullish_ff[i-1]
        
        # Only consider valid fractal levels (non-zero)
        if recent_bearish == 0:
            recent_bearish = None
        if recent_bullish == 0:
            recent_bullish = None
        
        # Regime filter: price above/below 1d EMA200
        bull_regime = close[i] > ema_200_1d_aligned[i]
        bear_regime = close[i] < ema_200_1d_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: Fractal breakout + volume + trend/regime alignment
        long_entry = False
        short_entry = False
        
        # Long: break above recent bearish fractal (resistance)
        if (recent_bearish is not None and 
            close[i] > recent_bearish and 
            vol_confirm[i] and 
            uptrend and 
            bull_regime and 
            position != 1):
            long_entry = True
        
        # Short: break below recent bullish fractal (support)
        if (recent_bullish is not None and 
            close[i] < recent_bullish and 
            vol_confirm[i] and 
            downtrend and 
            bear_regime and 
            position != -1):
            short_entry = True
        
        if long_entry:
            position = 1
            signals[i] = 0.25
        elif short_entry:
            position = -1
            signals[i] = -0.25
        # Exit: opposite fractal break or regime/trend change
        elif position == 1 and (recent_bullish is not None and close[i] < recent_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (recent_bearish is not None and close[i] > recent_bearish):
            position = 0
            signals[i] = 0.0
        elif position == 1 and (not bull_regime or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bear_regime or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals