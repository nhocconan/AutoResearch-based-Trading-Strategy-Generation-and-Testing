#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (14) + RSI (14) mean reversion with 12h EMA50 trend filter.
# In low volatility (choppy) markets, price tends to revert to mean. High Choppiness Index (>61.8) 
# indicates ranging conditions. Combine with RSI extremes (<30 for long, >70 for short) and 
# 12h EMA50 trend filter to avoid counter-trend trades. Low trade frequency (~15-25/year) 
# minimizes fee decay. Works in both bull and bear markets by adapting to ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA50 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 14-period Choppiness Index
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    
    # Calculate 14-period RSI
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        chop_val = chop[i]
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        
        # Chop filter: > 61.8 indicates ranging market (mean reversion)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long conditions: choppy market + RSI oversold + price above EMA (weak uptrend bias)
            if chop_filter and rsi_val < 30 and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short conditions: choppy market + RSI overbought + price below EMA (weak downtrend bias)
            elif chop_filter and rsi_val > 70 and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral or chop decreases
                if rsi_val > 50 or chop_val < 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral or chop decreases
                if rsi_val < 50 or chop_val < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_ChopRSI_MeanReversion_12hEMA50"
timeframe = "4h"
leverage = 1.0