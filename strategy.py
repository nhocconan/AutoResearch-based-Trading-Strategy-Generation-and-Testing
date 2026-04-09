#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI mean reversion + choppiness regime filter
# - Uses 1d primary timeframe with 1w HTF for trend context
# - KAMA(ER=10) identifies adaptive trend direction
# - RSI(14) < 30 for long, > 70 for short in ranging markets (CHOP > 61.8)
# - In trending markets (CHOP <= 61.8), follow KAMA direction
# - Volume confirmation: current volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_kama_rsi_chop_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Pre-compute KAMA (adaptive trend) on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smooth ER with fast=2/(2+2) and slow=2/(30+2)
    fast_sc = 2 / (2 + 2)
    slow_sc = 2 / (30 + 2)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14) on 1d
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:14] = 50.0
    
    # Pre-compute Choppiness Index (CHOP) on 1d
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_close - min_close) > 0, chop, 50.0)
    chop[:13] = 50.0  # Not enough data
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: reverse signal or stoploss
            if (close[i] < kama[i] and chop[i] <= 61.8) or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: reverse signal or stoploss
            if (close[i] > kama[i] and chop[i] <= 61.8) or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if volume_confirmed:
                if chop[i] > 61.8:  # Ranging market - mean reversion
                    if rsi[i] < 30 and close[i] > kama[i]:
                        position = 1
                        signals[i] = 0.25
                    elif rsi[i] > 70 and close[i] < kama[i]:
                        position = -1
                        signals[i] = -0.25
                else:  # Trending market - follow KAMA
                    if close[i] > kama[i] and rsi[i] > 50:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < kama[i] and rsi[i] < 50:
                        position = -1
                        signals[i] = -0.25
    
    return signals