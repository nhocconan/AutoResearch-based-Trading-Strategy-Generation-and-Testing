#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + RSI + 1d Bollinger Band Mean Reversion
# Long when 4h Choppiness Index > 61.8 (range) AND RSI(14) < 30 AND price < 1d Lower Bollinger Band
# Short when 4h Choppiness Index > 61.8 (range) AND RSI(14) > 70 AND price > 1d Upper Bollinger Band
# Exit when RSI returns to neutral (40-60) or opposite Bollinger band touch
# Uses 1d Bollinger Bands for mean reversion in ranging markets identified by 4h Choppiness
# Designed for low-volatility ranging markets (common in 2025 BTC/ETH) with mean reversion edge
# Target: 75-200 total trades over 4 years (19-50/year) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Indicators: Choppiness Index (14) and RSI (14) ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: (sum(ATR) / (max(high) - min(low))) * 100 * log10(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(14) * sum_atr / (highest_high - lowest_low + 1e-10)
    
    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Indicators: Bollinger Bands (20, 2) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        chop_val = chop[i]
        rsi_val = rsi[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when RSI returns to neutral (40-60) or price touches upper BB
            if rsi_val >= 40 and rsi_val <= 60:
                exit_signal = True
            elif price >= upper_bb_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral (40-60) or price touches lower BB
            if rsi_val >= 40 and rsi_val <= 60:
                exit_signal = True
            elif price <= lower_bb_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Chop > 61.8 (range) AND RSI < 30 (oversold) AND price < lower BB
            if (chop_val > 61.8 and rsi_val < 30 and price < lower_bb_val):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Chop > 61.8 (range) AND RSI > 70 (overbought) AND price > upper BB
            elif (chop_val > 61.8 and rsi_val > 70 and price > upper_bb_val):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Chop_RSI_BB_MeanRev_V1"
timeframe = "4h"
leverage = 1.0