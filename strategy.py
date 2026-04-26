#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_MeanReversion_VolumeFilter
Hypothesis: 4h KAMA trend direction (bullish/bearish) defines regime. In bullish regime, look for RSI < 30 (oversold) for long entries. In bearish regime, look for RSI > 70 (overbought) for short entries. Volume must be >1.5x 20-bar MA to confirm participation. Exits on opposite RSI extreme (RSI>70 for longs, RSI<30 for shorts) or opposite KAMA crossover. Designed for mean reversion within the trend, works in bull/bear via regime filter. Uses discrete position sizing (0.25) to minimize churn. Targets 20-50 trades/year.
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
    
    # Load 1d data ONCE before loop for HTF trend and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d KAMA for trend filter (ER=10, FAST=2, SLOW=30)
    close_1d = df_1d['close'].values
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Vectorized ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 10:
            dir = np.abs(close_1d[i] - close_1d[i-10])
            vol = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = dir / vol if vol != 0 else 0
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 4h RSI(14) for mean reversion signals
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        rsi_val = rsi[i]
        kama_val = kama_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > KAMA, bearish if price < KAMA
        bullish_regime = close_val > kama_val
        bearish_regime = close_val < kama_val
        
        # Entry conditions
        long_entry = bullish_regime and (rsi_val < 30) and vol_spike
        short_entry = bearish_regime and (rsi_val > 70) and vol_spike
        
        # Exit conditions: opposite RSI extreme or regime change
        exit_long = (rsi_val > 70) or (not bullish_regime and close_val < kama_val)
        exit_short = (rsi_val < 30) or (not bearish_regime and close_val > kama_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_KAMA_Trend_RSI_MeanReversion_VolumeFilter"
timeframe = "4h"
leverage = 1.0