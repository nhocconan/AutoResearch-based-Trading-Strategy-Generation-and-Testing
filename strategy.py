# 1d_KAMA_Trend_Filter_v1
# Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. Combined with 1-week trend filter and volume confirmation,
# this should capture strong trends while avoiding whipsaws in ranging markets.
# Works in both bull (captures trends) and bear (avoids false signals in ranges).

#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Using 10-period ER (Efficiency Ratio) and 2/30 smoothing constants
    er_period = 10
    fast_sc = 2 / (2 + 1)  # 2/(fast+1) where fast=2
    slow_sc = 2 / (30 + 1)  # 2/(slow+1) where slow=30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Pad the beginning with NaN
    change = np.concatenate([np.full(er_period, np.nan), change])
    volatility = np.concatenate([np.full(er_period, np.nan), volatility])
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, np.nan, dtype=float), where=volatility!=0)
    # Smooth the ER
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # Initialize
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate weekly EMA50 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align indicators to daily
    kama_aligned = align_htf_to_ltf(prices, None, kama)  # KAMA is already on daily index
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, weekly EMA, and volume MA
    start_idx = max(er_period + 5, 50, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price above KAMA with weekly uptrend and volume confirmation
            if (price > kama_aligned[i] and 
                ema_1w_aligned[i] > ema_1w_aligned[i-1] and  # Weekly uptrend
                vol_ratio > 1.3):
                signals[i] = size
                position = 1
            # Short: Price below KAMA with weekly downtrend and volume confirmation
            elif (price < kama_aligned[i] and 
                  ema_1w_aligned[i] < ema_1w_aligned[i-1] and  # Weekly downtrend
                  vol_ratio > 1.3):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below KAMA or weekly trend turns down
            if (price < kama_aligned[i] or 
                ema_1w_aligned[i] < ema_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above KAMA or weekly trend turns up
            if (price > kama_aligned[i] or 
                ema_1w_aligned[i] > ema_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0