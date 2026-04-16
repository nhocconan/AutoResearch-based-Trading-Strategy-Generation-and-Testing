#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for weekly pivot and bias) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly pivot from 1d data (using prior week's OHLC)
    # We'll calculate weekly pivot using 5-day lookback (approximation for weekly)
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # prior week high
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)   # prior week low
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(1)  # prior week close
    
    # Pivot point and support/resistance levels
    pivot = (high_5d + low_5d + close_5d) / 3.0
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    r2 = pivot + (high_5d - low_5d)
    s2 = pivot - (high_5d - low_5d)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # === 6h indicators ===
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        pivot_val = pivot_6h[i]
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        r2_val = r2_6h[i]
        s2_val = s2_6h[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 or RSI becomes overbought
            if (price < s1_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 or RSI becomes oversold
            if (price > r1_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above R1 AND RSI not overbought AND volume spike
                if (price > r1_val) and (rsi_val < 60) and (vol_ratio_val > 1.5):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 AND RSI not oversold AND volume spike
                elif (price < s1_val) and (rsi_val > 40) and (vol_ratio_val > 1.5):
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

name = "6h_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0