#1d_1w_kama_rsi_chop
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with KAMA trend filter, RSI momentum, and weekly Chop index regime filter.
# KAMA adapts to market efficiency - follows trends in trending markets, stays flat in ranging markets.
# RSI(14) > 55 for long, < 45 for short provides momentum confirmation.
# Weekly Chop index > 61.8 indicates ranging market (avoid trend following), < 38.2 indicates trending (follow trend).
# This combination should work in both bull (trend following) and bear (avoiding false signals in ranges) markets.
# Target: 20-80 total trades over 4 years (5-20/year) to minimize fee drag while capturing significant moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10-period)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for Chop index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Chop index (14-period)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max/min close over period
    max_close = pd.Series(close_1w).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop index
    chop = np.where((max_close - min_close) != 0, 
                    100 * np.log10(atr.sum() / (max_close - min_close)) / np.log10(14), 
                    50)
    # Handle initial values
    chop = np.concatenate([np.full(13, 50), chop])
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (Chop < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        # Entry conditions
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_long = rsi_aligned[i] > 55
        rsi_short = rsi_aligned[i] < 45
        
        long_entry = price_above_kama and rsi_long and trending_market
        short_entry = price_below_kama and rsi_short and trending_market
        
        # Exit conditions: opposite signal or chop indicates ranging market
        exit_long = position == 1 and (price_below_kama or not trending_market)
        exit_short = position == -1 and (price_above_kama or not trending_market)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop"
timeframe = "1d"
leverage = 1.0