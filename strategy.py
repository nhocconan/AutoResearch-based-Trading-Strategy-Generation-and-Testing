#!/usr/bin/env python3
# 6h_12h_turtle_soup_v1
# Hypothesis: Turtle Soup pattern on 6h chart with 12h trend filter. Looks for false breakouts
# of 6h Donchian(20) high/low that reverse quickly, indicating institutional stop hunting.
# Works in both bull and bear markets as it exploits mean reversion after false breakouts.
# Uses 12h EMA(50) for trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_turtle_soup_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter and stops
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    donch_high = np.full(len(high_6h), np.nan)
    donch_low = np.full(len(high_6h), np.nan)
    
    for i in range(20, len(high_6h)):
        donch_high[i] = np.max(high_6h[i-20:i])
        donch_low[i] = np.min(low_6h[i-20:i])
    
    # Align Donchian levels to 6h timeframe (wait for previous 6h bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.zeros_like(close_12h, dtype=float)
    ema_12h[0] = close_12h[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation - 24 period average (4 days of 6h bars)
    vol_ma_24 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.06 * close[i]  # ATR less than 6% of price
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_ok = volume[i] > vol_ma_24[i] * 1.3
        
        # Trend filter: price > 12h EMA for longs, price < 12h EMA for shorts
        trend_long = close[i] > ema_12h_aligned[i]
        trend_short = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (stop loss) or above Donchian high (profit target)
            if close[i] < donch_low_aligned[i] or close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (stop loss) or below Donchian low (profit target)
            if close[i] > donch_high_aligned[i] or close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Turtle Soup Long: False breakdown below Donchian low that reverses
            # Condition: price closes below Donchian low but then reverses back above it
            false_breakdown = (low[i] < donch_low_aligned[i]) and (close[i] > donch_low_aligned[i])
            
            # Turtle Soup Short: False breakout above Donchian high that reverses
            # Condition: price closes above Donchian high but then reverses back below it
            false_breakout = (high[i] > donch_high_aligned[i]) and (close[i] < donch_high_aligned[i])
            
            # Enter long on false breakdown with volume confirmation and trend filter
            if false_breakdown and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short on false breakout with volume confirmation and trend filter
            elif false_breakout and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals