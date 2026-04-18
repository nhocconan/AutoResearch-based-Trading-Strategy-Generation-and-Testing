#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter_V3
Hypothesis: Use daily Kaufman Adaptive Moving Average (KAMA) for trend direction, filtered by daily RSI(14) for momentum confirmation, with volume spike to confirm institutional participation. KAMA adapts to market noise, reducing whipsaws in chop while capturing trends. RSI filters extreme conditions to avoid counter-trend trades. Designed for low trade frequency (<15/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
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
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.subtract(close, np.roll(close, er_length)))
    vol = np.cumsum(change) - np.roll(np.cumsum(change), er_length)
    vol[vol == 0] = 1e-10  # avoid division by zero
    er = dir / vol
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: >1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Need KAMA warmup and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), volume spike
            if price > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), volume spike
            elif price < kama_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or RSI becomes overextended
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or RSI becomes oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter_V3"
timeframe = "1d"
leverage = 1.0