#!/usr/bin/env python3
name = "6H_Momentum_Confluence_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA200 for long-term trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Align weekly EMA200 to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for momentum and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily volume SMA(20) for volume filter
    vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Align daily volume SMA20 to 6h timeframe
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_sma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Long-term trend: price above/below weekly EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        # Momentum: RSI above 50 for bullish, below 50 for bearish
        bullish_momentum = rsi_1d_aligned[i] > 50
        bearish_momentum = rsi_1d_aligned[i] < 50
        # Volume filter: current volume > 1.5x daily volume SMA20
        volume_filter = volume[i] > vol_sma20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + bullish momentum + volume filter
            if uptrend and bullish_momentum and volume_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + bearish momentum + volume filter
            elif downtrend and bearish_momentum and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR momentum turns bearish
            if not uptrend or not bullish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR momentum turns bullish
            if not downtrend or not bearish_momentum:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals