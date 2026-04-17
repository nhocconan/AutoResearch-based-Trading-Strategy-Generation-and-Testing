#!/usr/bin/env python3
"""
4h_RSI_Trend_Pullback
Strategy: 4-hour RSI mean reversion with trend filter and volume confirmation.
Long: RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume > 1.3x average
Short: RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume > 1.3x average
Exit: RSI crosses back to 50 (mean reversion)
Position size: 0.25
Designed to capture mean reversion moves within established trends, working in both bull and bear markets.
Timeframe: 4h
"""

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
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_mean_revert = abs(rsi[i] - 50) < 10  # RSI near 50 for exit
        
        # Trend filter: price above/below EMA50
        price_above_ema = close[i] > ema50[i]
        price_below_ema = close[i] < ema50[i]
        
        if position == 0:
            # Long: RSI oversold + volume filter + price above EMA50 (uptrend)
            if rsi_oversold and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volume filter + price below EMA50 (downtrend)
            elif rsi_overbought and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI mean reversion (back to 50)
            if rsi_mean_revert:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI mean reversion (back to 50)
            if rsi_mean_revert:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Trend_Pullback"
timeframe = "4h"
leverage = 1.0