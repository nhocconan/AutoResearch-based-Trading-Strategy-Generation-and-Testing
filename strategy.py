#!/usr/bin/env python3
"""
1h_momentum_breakout_4h1d_trend_volume_v1
Hypothesis: On 1-hour timeframe, use momentum breakouts (price > high of last 20 bars) with trend filter from 4-hour EMA50 and 1-day EMA200, plus volume confirmation (>1.5x 20-period average). Enter long on breakout in uptrend (price > both EMAs), short on breakdown in downtrend (price < both EMAs). Exit on opposite breakout or trend reversal. Designed for moderate frequency (15-37 trades/year) to avoid excessive fees while capturing momentum. Uses multi-timeframe trend alignment to filter noise and works in both bull (buy strength in uptrend) and bear (sell weakness in downtrend) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_breakout_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    c4h = df_4h['close'].values
    ema4h_50 = pd.Series(c4h).ewm(span=50, adjust=False).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)
    
    # Get 1-day data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    c1d = df_1d['close'].values
    ema1d_200 = pd.Series(c1d).ewm(span=200, adjust=False).mean().values
    ema1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema1d_200)
    
    # Volume confirmation: 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if 4h or 1d EMA not available
        if np.isnan(ema4h_50_aligned[i]) or np.isnan(ema1d_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above both EMAs = uptrend, below both = downtrend
        uptrend = close[i] > ema4h_50_aligned[i] and close[i] > ema1d_200_aligned[i]
        downtrend = close[i] < ema4h_50_aligned[i] and close[i] < ema1d_200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit: price breaks below low of last 20 bars OR trend turns down
            if i >= 20:
                recent_low = np.min(low[i-20:i])
                trend_down = not uptrend  # either downtrend or mixed
                if close[i] < recent_low or trend_down:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above high of last 20 bars OR trend turns up
            if i >= 20:
                recent_high = np.max(high[i-20:i])
                trend_up = not downtrend  # either uptrend or mixed
                if close[i] > recent_high or trend_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need at least 20 periods for breakout calculation
            if i >= 20 and vol_confirm:
                recent_high = np.max(high[i-20:i])
                recent_low = np.min(low[i-20:i])
                
                # Long entry: price breaks above recent high in uptrend
                long_entry = close[i] > recent_high and uptrend
                # Short entry: price breaks below recent low in downtrend
                short_entry = close[i] < recent_low and downtrend
                
                if long_entry:
                    position = 1
                    signals[i] = 0.20
                elif short_entry:
                    position = -1
                    signals[i] = -0.20
    
    return signals