#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 1-day KAMA trend filter with 1-week momentum and volume confirmation
    # KAMA adapts to market noise - follows trend in trending markets, stays flat in choppy
    # Weekly momentum filter ensures we only trade with the higher timeframe trend
    # Volume spike confirms institutional participation
    # Target: 10-25 trades/year to minimize fee drag on daily timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA calculation (primary timeframe data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    # Efficiency Ratio: |change| / volatility
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1-week RSI for momentum filter
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.full_like(close_1w, np.nan)
    avg_loss = np.full_like(close_1w, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Volume spike filter on 1d
    vol_ma20 = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma20[i] = np.mean(volume_1d[i-20:i])
    vol_spike = volume_1d > 2.0 * vol_ma20
    
    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w, additional_delay_bars=0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma20_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + weekly RSI > 50 (bullish momentum) + volume spike
            if close[i] > kama_aligned[i] and rsi_1w_aligned[i] > 50 and vol_spike_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + weekly RSI < 50 (bearish momentum) + volume spike
            elif close[i] < kama_aligned[i] and rsi_1w_aligned[i] < 50 and vol_spike_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend reversal or momentum divergence
            if position == 1:
                if close[i] < kama_aligned[i] or rsi_1w_aligned[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i] or rsi_1w_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_1wRSI_Momentum_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0