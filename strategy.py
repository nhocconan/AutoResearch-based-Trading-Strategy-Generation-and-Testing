#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels (R1/S1) combined with RSI(14) momentum and volume spike.
# Weekly pivot levels act as major weekly support/resistance; breakouts with volume indicate strong momentum.
# RSI(14) filters for momentum alignment to avoid false breakouts.
# Designed for 6h timeframe to capture medium-term breakouts with low frequency.
# Entry: Long when close > R1 and RSI > 50 and volume spike; Short when close < S1 and RSI < 50 and volume spike.
# Exit: Opposite weekly pivot level touch (S1 for long exit, R1 for short exit) or RSI reversal.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_WeeklyPivot_R1S1_RSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels (based on prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC for pivot calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly pivot point and R1/S1 levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align to 6h timeframe (waits for prior week close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with bullish momentum and volume
            if (close[i] > r1_aligned[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with bearish momentum and volume
            elif (close[i] < s1_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches S1 or RSI turns bearish
            if (close[i] < s1_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches R1 or RSI turns bullish
            if (close[i] > r1_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals