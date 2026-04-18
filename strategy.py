#!/usr/bin/env python3
"""
1d_Momentum_Swing_Signal
Strategy: Daily momentum swing using 21-period RSI and 50-period SMA crossover for trend direction.
Enters long when RSI crosses above 50 with price above SMA50, enters short when RSI crosses below 50 with price below SMA50.
Exits on opposite signal or weekly trend reversal. Designed for low trade frequency (~10-20 trades/year) with strong signal quality.
Works in bull/bear via SMA50 trend filter and RSI momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA50 to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate daily RSI(21)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/21, adjust=False, min_periods=21).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/21, adjust=False, min_periods=21).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily SMA50 for entry filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # RSI momentum signals
        rsi_cross_up = (rsi[i-1] < 50) and (rsi[i] >= 50)
        rsi_cross_down = (rsi[i-1] > 50) and (rsi[i] <= 50)
        
        if position == 0:
            # Long: weekly uptrend + RSI cross above 50 + price above daily SMA50
            if weekly_uptrend and rsi_cross_up and close[i] > sma_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI cross below 50 + price below daily SMA50
            elif weekly_downtrend and rsi_cross_down and close[i] < sma_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend reversal or RSI cross below 50
            if not weekly_uptrend or rsi_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend reversal or RSI cross above 50
            if not weekly_downtrend or rsi_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Momentum_Swing_Signal"
timeframe = "1d"
leverage = 1.0