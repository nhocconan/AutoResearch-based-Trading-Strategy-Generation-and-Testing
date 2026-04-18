#!/usr/bin/env python3
"""
4h KAMA + RSI + Chop Regime + Volume Spike
Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing smooth trend direction.
Long when price > KAMA, RSI > 50, Chop > 61.8 (trending), and volume spike.
Short when price < KAMA, RSI < 50, Chop > 61.8, and volume spike.
Uses 1-day ATR for volatility filter to avoid choppy markets.
Designed for low trade frequency with adaptive trend following.
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
    
    # Get daily data for 1-day ATR volatility filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # KAMA (2, 10, 30) - ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(close - np.roll(close, 10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # 10-period volatility
    # Calculate volatility over 10 periods
    vol_10 = np.zeros_like(close)
    for i in range(10, len(close)):
        vol_10[i] = np.sum(np.abs(np.diff(close[i-10:i+1], prepend=close[i-10])))
    er = np.where(vol_10 != 0, change / vol_10, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2  # SC = [ER*(fastest - slowest) + slowest]^2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    high_low_range = pd.Series(high - low).rolling(window=14, min_periods=14).max().values - \
                     pd.Series(high - low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / high_low_range) / np.log10(14)
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (Chop > 61.8) and sufficient volatility (ATR > 0.5% of price)
        trending = chop[i] > 61.8
        sufficient_vol = atr_14_1d_aligned[i] > (0.005 * close[i])
        
        if not (trending and sufficient_vol):
            signals[i] = 0.0
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        above_kama = price > kama[i]
        below_kama = price < kama[i]
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, volume spike
            if (above_kama and rsi_bullish and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, volume spike
            elif (below_kama and rsi_bearish and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price < KAMA or RSI < 40
            if below_kama or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price > KAMA or RSI > 60
            if above_kama or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_RSI_Chop_VolumeSpike"
timeframe = "4h"
leverage = 1.0