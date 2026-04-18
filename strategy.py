#!/usr/bin/env python3
"""
6h_RSI_MeanReversion_BollingerBand_v1
Strategy: 6h RSI mean reversal with Bollinger Band support/resistance and volume confirmation.
Long: RSI < 30 and price touches lower BB with volume > 1.3x average.
Short: RSI > 70 and price touches upper BB with volume > 1.3x average.
Exit: RSI crosses 50 (mean reversion complete) or volatility expansion.
Uses 1d trend filter: only trade in direction of daily EMA50 to avoid counter-trend in strong trends.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
Works in ranging markets; trend filter avoids major losses in trends.
"""

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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Bollinger Bands(20,2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_upper = bb_upper.values
    bb_lower = bb_lower.values
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: only long in uptrend, only short in downtrend
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: RSI oversold + touches lower BB + volume + uptrend
            if (rsi[i] < 30 and close[i] <= bb_lower[i] and vol_confirm and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + touches upper BB + volume + downtrend
            elif (rsi[i] > 70 and close[i] >= bb_upper[i] and vol_confirm and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses 50 or volatility expansion (price touches upper BB)
            if rsi[i] >= 50 or close[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses 50 or volatility expansion (price touches lower BB)
            if rsi[i] <= 50 or close[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_MeanReversion_BollingerBand_v1"
timeframe = "6h"
leverage = 1.0