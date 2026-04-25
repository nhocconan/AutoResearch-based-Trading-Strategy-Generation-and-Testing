#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v3
Hypothesis: Use 1d KAMA for primary trend direction, RSI(14) for momentum confirmation, and Choppiness Index for regime filtering.
- Long: price > KAMA(10) AND RSI(14) > 50 AND CHOP(14) < 38.2 (trending regime)
- Short: price < KAMA(10) AND RSI(14) < 50 AND CHOP(14) < 38.2 (trending regime)
- Exit: trend reversal (price crosses KAMA) or choppy regime (CHOP > 61.8)
- Volume confirmation: require volume > 1.5x 20-period average to avoid false signals
- Position size: 0.25. Target: 30-100 total trades over 4 years (7-25/year).
- Works in both bull and bear: KAMA adapts to market noise, RSI confirms momentum, chop filter avoids ranging markets.
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
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d KAMA for primary trend
    # KAMA parameters: ER period=10, fast=2, slow=30
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close[0]]  # seed
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate 1d RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(1), n) / (max(high, n) - min(low, n))) / log10(n)
    atr_period = 14
    tr1 = high - low
    tr2 = abs(high - np.roll(close, 1))
    tr3 = abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean()
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / (max_high - min_low)) / np.log10(14)
    chop = chop.values
    
    # Volume spike confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA(10), RSI(14), CHOP(14), volume MA(20)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend using EMA50
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Entry logic: trade in direction of 1w trend with momentum and regime filters
            long_setup = (close[i] > kama[i]) and (rsi[i] > 50) and (chop[i] < 38.2) and htf_1w_bullish and volume_spike[i]
            short_setup = (close[i] < kama[i]) and (rsi[i] < 50) and (chop[i] < 38.2) and htf_1w_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal, momentum loss, or choppy regime
            exit_signal = (close[i] < kama[i]) or (rsi[i] < 40) or (chop[i] > 61.8)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal, momentum loss, or choppy regime
            exit_signal = (close[i] > kama[i]) or (rsi[i] > 60) or (chop[i] > 61.8)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v3"
timeframe = "1d"
leverage = 1.0