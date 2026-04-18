#!/usr/bin/env python3
"""
12h KAMA Direction + Daily RSI + Volume Spike
Strategy: Go long when KAMA is rising and RSI < 30 (oversold) with volume spike,
          short when KAMA is falling and RSI > 70 (overbought) with volume spike.
          Uses daily RSI for mean-reversion edge and KAMA for trend direction.
          Designed for low trade frequency with mean-reversion in ranging markets
          and trend following in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate KAMA on 12h close
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np.sum, 'axis') else np.sum(np.abs(np.diff(close, prepend=close[0])))
    # Correct volatility calculation for 10-period
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, volume spike
            if i > 1 and kama_val > kama[i-1] and rsi_val < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, volume spike
            elif i > 1 and kama_val < kama[i-1] and rsi_val > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: KAMA falling or RSI overbought
            if i > 0 and (kama_val < kama[i-1] or rsi_val > 70):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: KAMA rising or RSI oversold
            if i > 0 and (kama_val > kama[i-1] or rsi_val < 30):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_RSI_VolumeSpike"
timeframe = "12h"
leverage = 1.0