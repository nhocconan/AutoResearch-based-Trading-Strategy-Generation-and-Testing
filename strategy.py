#!/usr/bin/env python3
"""
1h_Stochastic_4hTrend_RSIFilter
Hypothesis: In 1h timeframe, use 4h trend filter (EMA50) with Stochastic oscillator for mean-reversion entries and RSI(14) for momentum confirmation. This combination reduces whipsaws in ranging markets while capturing trend continuations. Works in both bull and bear markets by adapting to regime via 4h trend filter. Targets 15-37 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Stochastic(14,3,3) on 1h
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    # Avoid division by zero
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Calculate RSI(14) on 1h for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Default to neutral when no loss
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(d_percent[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Stochastic signals
        stoch_oversold = d_percent[i] < 20
        stoch_overbought = d_percent[i] > 80
        stoch_cross_up = (d_percent[i-1] < 20) and (d_percent[i] >= 20) and (d_percent[i] > d_percent[i-1])
        stoch_cross_down = (d_percent[i-1] > 80) and (d_percent[i] <= 80) and (d_percent[i] < d_percent[i-1])
        
        # RSI momentum confirmation
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry logic: mean reversion in direction of trend
        long_entry = stoch_cross_up and uptrend and rsi_bullish
        short_entry = stoch_cross_down and downtrend and rsi_bearish
        
        # Exit logic: opposite stochastic cross or trend change
        long_exit = stoch_cross_down or (not uptrend)
        short_exit = stoch_cross_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Stochastic_4hTrend_RSIFilter"
timeframe = "1h"
leverage = 1.0