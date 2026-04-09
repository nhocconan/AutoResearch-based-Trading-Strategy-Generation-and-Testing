#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with 1w EMA200 trend filter and volume confirmation
# Uses Camarilla levels from 1d data: long at L3 with bullish 1w trend, short at H3 with bearish 1w trend
# 1w EMA200 filter ensures trades align with weekly trend (more stable than 1d)
# Volume confirmation reduces false breakouts
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull/bear: EMA200 adapts to trend, Camarilla provides mean-reversion structure

name = "1d_1w_camarilla_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla formula: 
    # H4 = close + 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # L4 = close - 1.1*(high-low)*1.1/2
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    H3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    H4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Align 1d Camarilla levels to 1d timeframe (no shift needed as they're based on prev day)
    H3_1d = align_htf_to_ltf(prices, df_1d, H3)
    L3_1d = align_htf_to_ltf(prices, df_1d, L3)
    H4_1d = align_htf_to_ltf(prices, df_1d, H4)
    L4_1d = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate 1w EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1w EMA200 to 1d timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period average volume for volume confirmation (1d volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H3_1d[i]) or np.isnan(L3_1d[i]) or np.isnan(H4_1d[i]) or 
            np.isnan(L4_1d[i]) or np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR trend turns bearish (price < EMA200)
            if close[i] < L3_1d[i] or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR trend turns bullish (price > EMA200)
            if close[i] > H3_1d[i] or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long entry: price closes below L3 (mean reversion) AND price > 1w EMA200 (bullish trend)
                if close[i] < L3_1d[i] and close[i] > ema_200_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes above H3 (mean reversion) AND price < 1w EMA200 (bearish trend)
                elif close[i] > H3_1d[i] and close[i] < ema_200_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals