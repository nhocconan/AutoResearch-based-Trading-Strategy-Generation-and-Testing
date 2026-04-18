#!/usr/bin/env python3
"""
1d_1w_Kelly_Fractional_Kelly_Strategy
Hypothesis: Use weekly price action to determine market regime (bull/bear/range) and apply fractional Kelly criterion for position sizing on daily timeframe. In bull regime (price > weekly SMA50), go long with Kelly size based on daily RSI mean reversion. In bear regime (price < weekly SMA50), go short with Kelly size. In range (price near weekly SMA50), stay flat. This adapts position size to edge strength, reducing exposure in uncertain markets and maximizing growth in trending ones. Targets 10-20 trades/year with proper risk control.
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
    
    # Get weekly data for regime determination
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly SMA50 for regime
    weekly_sma50 = np.full_like(weekly_close, np.nan)
    for i in range(50, len(weekly_close)):
        weekly_sma50[i] = np.mean(weekly_close[i-50:i])
    
    # Align weekly SMA50 to daily
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Calculate daily RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Kelly fraction components
    # Win probability based on RSI extremes (oversold/overbought)
    # In bull regime: long when RSI < 30 (oversold)
    # In bear regime: short when RSI > 70 (overbought)
    # Win rate assumed 60% for extreme RSI readings
    win_prob = 0.60
    # Average win/loss ratio based on ATR
    # Calculate daily ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full_like(close, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Use ATR as proxy for average win/loss
    # In mean reversion, target 1x ATR profit, 0.5x ATR loss
    avg_win = atr
    avg_loss = 0.5 * atr
    win_loss_ratio = np.where(avg_loss != 0, avg_win / avg_loss, 2.0)
    
    # Kelly fraction: f = (bp - q) / b where b = win/loss ratio, p = win prob, q = loss prob
    kelly_fraction = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
    kelly_fraction = np.clip(kelly_fraction, 0, 0.5)  # Cap at 50%, use half-Kelly for safety
    
    signals = np.zeros(n)
    
    start_idx = max(50, 14)  # need weekly SMA50 and daily RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_sma50_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_sma = weekly_sma50_aligned[i]
        
        # Determine regime: bull if price > weekly SMA50, bear if price < weekly SMA50
        # Add hysteresis to prevent whipsaw: 1% buffer
        bull_threshold = weekly_sma * 1.01
        bear_threshold = weekly_sma * 0.99
        
        if price > bull_threshold:
            # Bull regime: look for long opportunities on RSI oversold
            if rsi[i] < 30:
                kelly = kelly_fraction[i]
                signals[i] = min(kelly, 0.30)  # Cap position size at 30%
            else:
                signals[i] = 0.0
        elif price < bear_threshold:
            # Bear regime: look for short opportunities on RSI overbought
            if rsi[i] > 70:
                kelly = kelly_fraction[i]
                signals[i] = -min(kelly, 0.30)  # Cap position size at 30%
            else:
                signals[i] = 0.0
        else:
            # Range regime: stay flat
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Kelly_Fractional_Kelly_Strategy"
timeframe = "1d"
leverage = 1.0