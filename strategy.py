#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing reliable trend direction.
Combined with RSI for momentum confirmation and Choppiness Index to avoid false signals in ranging markets.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag and improve generalization.
Works in both bull (trend following) and bear (avoids false breakouts via chop filter).
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
    
    # Calculate KAMA (ER=10, fast=2, slow=30) - trend indicator
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    abs_change = np.abs(np.diff(close, k=1))  # 1-period absolute change
    er = np.zeros(n)
    er[10:] = change[10:] / (np.abs(np.diff(close, k=1))[10:].cumsum() - np.abs(np.diff(close, k=1))[0:].cumsum() + 1e-10)
    er = np.where(np.isnan(er), 0, er)
    er = np.clip(er, 0, 1)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) - momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) - regime filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), chop)
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14), volume avg (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: price above KAMA, RSI > 50, not choppy, volume confirmation, weekly uptrend
            if (close_val > kama_val and 
                rsi_val > 50 and 
                chop_val < 61.8 and 
                vol_conf and 
                close_val > ema50):
                signals[i] = size
                position = 1
            # Short conditions: price below KAMA, RSI < 50, not choppy, volume confirmation, weekly downtrend
            elif (close_val < kama_val and 
                  rsi_val < 50 and 
                  chop_val < 61.8 and 
                  vol_conf and 
                  close_val < ema50):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0