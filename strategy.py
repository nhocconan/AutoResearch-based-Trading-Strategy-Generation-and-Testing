#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation
# Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND price > weekly EMA50 AND volume > 1.5x 1d average volume
# Short when price breaks below Kumo AND Tenkan < Kijun (bearish TK cross) AND price < weekly EMA50 AND volume > 1.5x 1d average volume
# Ichimoku provides dynamic support/resistance with forward-looking cloud
# Weekly EMA50 ensures alignment with longer-term trend
# Volume confirmation adds conviction to breakouts
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Ichimoku Cloud ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === 1d Volume Confirmation ===
    # 24 periods of 1h = 1d (using 6h data: 4 periods per day)
    vol_ma_1d = pd.Series(volume).rolling(window=4, min_periods=4).mean().values  # 4 periods of 6h = 1d
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
        
        # Kumo (cloud) boundaries
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_kumo = price > upper_kumo
        price_below_kumo = price < lower_kumo
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Kumo AND TK bullish AND price > weekly EMA50 AND volume confirmation
            if price_above_kumo and tk_bullish and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below Kumo AND TK bearish AND price < weekly EMA50 AND volume confirmation
            elif price_below_kumo and tk_bearish and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC ===
        elif position == 1:  # Long position
            # Exit when price breaks below Kumo OR TK turns bearish
            if price < lower_kumo or not tk_bullish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Kumo OR TK turns bullish
            if price > upper_kumo or tk_bullish:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
        
        # Hold flat
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_KumoBreak_TKCross_1wEMA50_Volume1.5x"
timeframe = "6h"
leverage = 1.0