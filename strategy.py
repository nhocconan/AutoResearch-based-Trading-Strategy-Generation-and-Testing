#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d volume confirmation and 1d RSI filter
# - KAMA(14,2,30) for trend direction: long when KAMA rising, short when falling
# - 1d volume > 1.5x 20-period average for conviction
# - 1d RSI(14) filter: avoid extremes (RSI between 30 and 70) to prevent counter-trend entries
# - Designed to work in both bull and bear markets by following trend with volume confirmation
# - Target: 25-40 trades/year to avoid excessive fee drift

name = "4h_KAMA_1dVolume_RSIFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA(14,2,30) on 4h close
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = change / (volatility + 1e-10)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    kama_diff = np.diff(kama_vals, prepend=kama_vals[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for KAMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_diff[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x scaled 1d average volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = 30 <= rsi_1d_aligned[i] <= 70
        
        if position == 0:
            # Look for long entry: rising KAMA + volume + RSI filter
            if kama_diff[i] > 0 and volume_filter and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: falling KAMA + volume + RSI filter
            elif kama_diff[i] < 0 and volume_filter and rsi_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on falling KAMA
            if kama_diff[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on rising KAMA
            if kama_diff[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals