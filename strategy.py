#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Volume_Trend
Hypothesis: On 4h timeframe, enter long when price breaks above Keltner upper band with 12h volume confirmation and bullish 12h trend; short when price breaks below Keltner lower band with volume and bearish 12h trend. Keltner Channels adapt to volatility better than Bollinger Bands in crypto markets, reducing whipsaws. The 12h trend filter ensures alignment with higher timeframe momentum, while volume confirmation avoids low-conviction breakouts. Designed for 15-35 trades per year to minimize fee drag and work in both bull/bear regimes via trend alignment.
"""

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
    volume = prices['volume'].values
    
    # === 12h data for volume and trend ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Keltner Channel (20, 2) on 4h data
    # Middle = EMA(20), Width = ATR(20) * 2
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr20 = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    kc_upper = ema20 + 2 * atr20
    kc_lower = ema20 - 2 * atr20
    
    # 12h volume average (20-period) for confirmation
    vol_avg20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA20, ATR20, and 12h indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema20[i]) or np.isnan(atr20[i]) or 
            np.isnan(vol_avg20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h volume
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        
        # Volume filter: current volume > 1.8x 20-period average
        vol_filter = vol_12h_current > 1.8 * vol_avg20_12h_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        above_trend = close[i] > ema50_12h_aligned[i]
        below_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Keltner upper + volume + above 12h trend
            if close[i] > kc_upper[i] and vol_filter and above_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Keltner lower + volume + below 12h trend
            elif close[i] < kc_lower[i] and vol_filter and below_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal when price crosses EMA20 (middle)
        elif position == 1:
            if close[i] < ema20[i]:  # price crosses below middle = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > ema20[i]:  # price crosses above middle = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0