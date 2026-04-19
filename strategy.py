#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Volume and ATR Filter
# Weekly pivot levels (from weekly high/low/close) provide strong support/resistance
# Price breaking above weekly R1 or below weekly S1 with volume confirmation
# indicates institutional interest and trend continuation
# ATR filter avoids choppy markets; only trade when volatility is expanding
# Weekly timeframe ensures relevance for 6h chart; reduces noise
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
# Works in bull (breakouts continuation) and bear (breakdowns continuation) markets
name = "6h_WeeklyPivot_Breakout_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # Support/resistance: R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # ATR filter: only trade when volatility is expanding (ATR > 20-period average)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR with 20-period
    atr = np.zeros(n)
    atr[19] = tr[:20].mean()
    for i in range(20, n):
        atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_expanding = atr > (atr_ma * 1.0)  # ATR above its average
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Ensure enough data for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(atr_ma[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + volume + volatility expanding
            if (close[i] > weekly_r1_aligned[i] and 
                volume_confirm[i] and 
                volatility_expanding[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + volume + volatility expanding
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_confirm[i] and 
                  volatility_expanding[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly pivot OR volatility contracts
            if (close[i] < weekly_pivot_aligned[i]) or (not volatility_expanding[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly pivot OR volatility contracts
            if (close[i] > weekly_pivot_aligned[i]) or (not volatility_expanding[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals