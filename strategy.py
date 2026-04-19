#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter (EMA) and 1d regime filter (Chop).
# Long when price > 4h EMA20 AND Chop < 50 (trending), with 1h pullback entry.
# Short when price < 4h EMA20 AND Chop < 50, with 1h bounce entry.
# Uses Chop to avoid ranging markets where trend following fails.
# Target: 60-150 total trades over 4 years (15-37/year).
name = "1h_4hEMA_1dChop_Pullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for Chop regime (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop on 1d (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr) / (max(high) - min(low))) / log10(14)
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop[np.isnan(chop)] = 100  # Default to ranging when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h EMA for pullback entry
    ema_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_4h_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(ema_1h[i]):
            signals[i] = 0.0
            continue
            
        # Check regime: trending (Chop < 50)
        trending = chop_aligned[i] < 50
        
        if not trending or not in_session[i]:
            # Exit position if not trending or outside session
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Trend direction from 4h EMA
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        if position == 0:
            # Long: pullback to 1h EMA in uptrend
            if trend_up and close[i] <= ema_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: bounce to 1h EMA in downtrend
            elif trend_down and close[i] >= ema_1h[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit on trend reversal or overextension
            if not trend_up or close[i] > ema_1h[i] * 1.02:  # 2% above EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit on trend reversal or overextension
            if not trend_down or close[i] < ema_1h[i] * 0.98:  # 2% below EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals