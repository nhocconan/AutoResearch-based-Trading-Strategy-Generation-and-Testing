#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Long when price closes above Donchian(20) upper band AND price > 1w EMA50 AND volume > 1.5x 30d average volume
# Short when price closes below Donchian(20) lower band AND price < 1w EMA50 AND volume > 1.5x 30d average volume
# ATR trailing stop (2.5x ATR) to manage risk
# Donchian breakout provides clear breakout signals with defined risk
# 1w EMA50 ensures alignment with long-term trend
# Volume confirmation adds conviction to breakouts
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === 1d Donchian channels (20-period) ===
    # Calculate highest high and lowest low of past 20 days
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 30d Volume average for confirmation ===
    vol_avg_30d = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # === 1d ATR for trailing stop (15-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and exit levels
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_avg_30d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        vol_confirm = volume[i] > vol_avg_30d[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit if price breaks below lower Donchian band
            if price < lower_band:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if price breaks above upper Donchian band
            if price > upper_band:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price closes above Donchian upper band AND price > EMA50 AND volume confirmation
            if price > upper_band and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price closes below Donchian lower band AND price < EMA50 AND volume confirmation
            elif price < lower_band and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume1.5x"
timeframe = "1d"
leverage = 1.0