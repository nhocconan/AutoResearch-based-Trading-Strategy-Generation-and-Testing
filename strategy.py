#!/usr/bin/env python3
"""
1d_Adaptive_Kelly_RSI_Extreme_Volatility
Hypothesis: Daily RSI extremes (oversold/overbought) combined with volatility expansion signals and adaptive position sizing based on Kelly criterion.
Works in bull markets by buying oversold dips in uptrends, and in bear markets by selling overbought rallies in downtrends.
Volatility filter ensures trades occur during significant market moves, reducing whipsaw.
Adaptive scaling reduces position size during low volatility periods to minimize drawdown.
Target: 15-25 trades/year to minimize fee drag while capturing high-probability mean reversions.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily ATR(14) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period average ATR
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma_50 + 1e-10)
    
    # Volatility expansion signal: ATR ratio > 1.2
    vol_expansion = atr_ratio > 1.2
    
    # RSI extreme conditions
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Trend filter from weekly EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: RSI extreme in direction of trend with volatility expansion
        long_entry = rsi_oversold[i] and uptrend[i] and vol_expansion[i]
        short_entry = rsi_overbought[i] and downtrend[i] and vol_expansion[i]
        
        # Exit logic: RSI returns to neutral zone or trend changes
        long_exit = rsi[i] > 50 or not uptrend[i]
        short_exit = rsi[i] < 50 or not downtrend[i]
        
        if long_entry and position <= 0:
            # Adaptive sizing: base size 0.25, scaled by volatility (capped)
            vol_scale = min(atr_ratio[i] / 1.5, 1.5)  # Scale with vol, max 1.5x
            size = 0.25 * vol_scale
            size = min(max(size, 0.10), 0.35)  # Clamp between 0.10 and 0.35
            signals[i] = size
            position = 1
        elif short_entry and position >= 0:
            vol_scale = min(atr_ratio[i] / 1.5, 1.5)
            size = 0.25 * vol_scale
            size = min(max(size, 0.10), 0.35)
            signals[i] = -size
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
                signals[i] = 0.25  # Base size when holding
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Adaptive_Kelly_RSI_Extreme_Volatility"
timeframe = "1d"
leverage = 1.0