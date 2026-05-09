#!/usr/bin/env python3
# Hypothesis: 1d timeframe with weekly VWAP as support/resistance and daily RSI for mean reversion.
# Uses weekly VWAP (volume-weighted average price) for structural support/resistance.
# Daily RSI(14) < 30 for long entries, > 70 for short entries.
# Weekly VWAP provides robust levels that work in both bull and bear markets.
# Daily RSI helps avoid buying strength or selling weakness.
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25.

name = "1d_VWAP_Weekly_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly VWAP from previous week
    # VWAP = sum(price * volume) / sum(volume) for the week
    # We'll use typical price * volume for simplicity
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume  # typical price * volume
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    # Need to sum over the week - we'll use 5 trading days approximation
    # For simplicity, we'll use the weekly typical price and volume from the weekly data
    weekly_typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_tpv = weekly_typical_price * df_1w['volume']
    weekly_vwap = weekly_tpv.cumsum() / df_1w['volume'].cumsum()
    # Handle division by zero
    weekly_vwap = weekly_vwap.replace([np.inf, -np.inf], np.nan)
    
    # Align weekly VWAP to daily timeframe with proper delay
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap.values)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_vwap_aligned[i]) or np.isnan(rsi.iloc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below weekly VWAP + RSI < 30 (oversold)
            if close[i] < weekly_vwap_aligned[i] and rsi.iloc[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price above weekly VWAP + RSI > 70 (overbought)
            elif close[i] > weekly_vwap_aligned[i] and rsi.iloc[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back above weekly VWAP or RSI > 50
            if close[i] > weekly_vwap_aligned[i] or rsi.iloc[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back below weekly VWAP or RSI < 50
            if close[i] < weekly_vwap_aligned[i] or rsi.iloc[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals