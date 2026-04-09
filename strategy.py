#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with weekly trend filter
# Uses 1d Camarilla levels (H4/L4) for mean reversion entries
# Filters by 1w trend: only take long when price > 1w EMA20, short when price < 1w EMA20
# Volume confirmation: current volume > 1.5x 24-period average
# Target: 15-30 trades/year per symbol to minimize fee drag
# Works in bull/bear: mean reversion in range, trend filter avoids counter-trend trades

name = "6h_1w_camarilla_reversal_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + range_ * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(df_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Align 1w EMA20 to 6h timeframe
    ema_20_1w_6h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 24-period average on 6h (4 days)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_6h[i]) or 
            np.isnan(camarilla_l4_6h[i]) or 
            np.isnan(ema_20_1w_6h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla H4 or trend turns bearish
            if close[i] >= camarilla_h4_6h[i] or close[i] < ema_20_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla L4 or trend turns bullish
            if close[i] <= camarilla_l4_6h[i] or close[i] > ema_20_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches Camarilla L4 with volume confirmation, in bullish trend
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] <= camarilla_l4_6h[i] and 
                vol_ratio > 1.5 and 
                close[i] > ema_20_1w_6h[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches Camarilla H4 with volume confirmation, in bearish trend
            elif (close[i] >= camarilla_h4_6h[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < ema_20_1w_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals